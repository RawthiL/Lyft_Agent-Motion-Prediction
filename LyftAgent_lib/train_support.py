#!/usr/bin/python
# -*- coding: iso-8859-15 -*-


import os, sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0])
import topologies as lyl_nn

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from l5kit.evaluation import metrics





###############################################################################
# ------------------------ CUSTOM TRAIN FUNCTIONS --------------------------- #
###############################################################################

@tf.function
def generator_train_step(img_t, hist_t, target_path, HistAvail, TargetAvail,
                         ImageEncModel, HistEncModel, PathDecModel, 
                         optimizer_gen, 
                         gen_loss,
                         initial_hidden_state,
                         forwardpass_use,
                         stepsInfer = None,
                         use_teacher_force=True, 
                         teacher_force_weight = tf.constant(1.0, dtype=tf.float32)):
    
    '''
    Calculate the mean L2 loss of the prediction.
    
    Arguments:
    img_t --- Multi-channel map input tensor.
    hist_t --- Path history tensor.
    target_path --- Target/Real path.
    HistAvail --- Number valid history points. 
    TargetAvail --- Number valid target/real points. 
    ImageEncModel --- Model used to encode the input image (img_t)
    HistEncModel --- Model used to encode the input history (hist_t)
    PathDecModel --- Model used to produce the requested predictions.
    optimizer_gen --- Keras optimizer object.
    gen_loss --- Keras metric object.
    initial_hidden_state --- Initial hidden state of the decoder model.
    use_teacher_force --- Use a linear interpolation of the target path and the last predicted path.
    teacher_force_weight --- Fraction of the target path used in the linear interpolation between target and predicted.
    '''
    
    # Get number of steps to generate
    if stepsInfer == None:
        stepsInfer = target_path.shape[1]
    
    # Start gradient tape
    with tf.GradientTape() as gen_tape:
              
        # Get predicted paths
        thisPath = forwardpass_use(img_t, hist_t, HistAvail, stepsInfer, 
                                      ImageEncModel, HistEncModel, PathDecModel,
                                      thisHiddenState = initial_hidden_state, 
                                      use_teacher_force = use_teacher_force, 
                                      teacher_force_weight = teacher_force_weight,
                                      target_path = target_path)      
        # Compute predicted path loss
        gen_L2_loss = calc_loss(thisPath,target_path[:,:stepsInfer,:], TargetAvail[:,:stepsInfer])
        
    # Get list of trainable variables
    train_vars = ImageEncModel.trainable_variables+ \
                 HistEncModel.trainable_variables+ \
                 PathDecModel.trainable_variables
    # Compute gradient
    grad_gen = gen_tape.gradient(target=gen_L2_loss, 
                                 sources=train_vars)
    # Apply update
    optimizer_gen.apply_gradients(zip(grad_gen, train_vars))
    
    # Save loss
    gen_loss(tf.reduce_mean(gen_L2_loss))
        
    return

def get_teacher_force_weight(tf_list, tf_lims, epoch, tf_weight_act, linearize=False):
    '''
    Get the teacher force weight given the epoch and the weight planning.
    
    Arguments:
    tf_list --- List of teacher force weights.
    tf_lims --- List of teacher force limit epochs for each weight.
    epoch --- Current train epoch.
    tf_weight_act --- Current teacher force weight.
    linearize --- Linearize the teacher force weight between the given points.
        
    Outputs:
    tf_weight_new --- New teacher force weight.
    '''
    
    # Get current weight
    tf_lims = np.array(tf_lims)
    
    if linearize:
        tf_weight_new = np.interp(epoch, tf_lims, tf_list)
    else:
        tf_weight_new = tf_list[np.squeeze(np.argwhere(tf_lims>=epoch)[0])]
            
    # If the value is new, inform
    if tf_weight_act != tf_weight_new:
        print('Teacher force weight set to: %g'%tf_weight_new)
    
    return tf_weight_new
   
    
def update_lr(lr_list, lr_lims, epoch, optimizer):
    '''
    Set the learning rate given the epoch and the learning rate planning.
       
    Arguments:
    lr_list --- List of learning rates.
    lr_lims --- List of learning rate limit epochs.
    epoch --- Current train epoch.
    optimizer --- Keras optimizer.
    '''
    
    # Get new learning rate
    lr_new = lr_list[np.squeeze(np.argwhere(np.array(lr_lims) >= epoch)[0])]
    # Update and inform if needed
    if not np.isclose(lr_new, optimizer.learning_rate.numpy(), rtol = 1e-8):
        K.set_value(optimizer.learning_rate, lr_new)
        print('Learning rate set to: %g'%optimizer.learning_rate.numpy())
        
    return


def get_future_steps_train(fs_list, fs_lims, epoch, fs_act):
    '''
    Get the number of future steps to train on given the epoch and the planning.
    
    Arguments:
    fs_list --- List of future steps.
    fs_lims --- List of limit epochs.
    epoch --- Current train epoch.
    fs_act --- Current future steps weight.
        
    Outputs:
    fs_new --- New future steps to train on.
    '''
    
    # Get current weight
    fs_lims = np.array(fs_lims)
    
    fs_new = fs_list[np.squeeze(np.argwhere(fs_lims>=epoch)[0])]
            
    # If the value is new, inform
    if fs_act != fs_new:
        print('Training on %d future steps.'%fs_new)
    
    return fs_new


###############################################################################
# ------------------------ VALIDATION FUNCTIONS ----------------------------- #
###############################################################################

def validate_model(tf_validation_dataset, ImageEncModel, HistEncModel, PathDecModel, forwardpass_use,
                   steps_validate = None):
    '''
    Validate the generator model using the validation dataset.
    
    Arguments:
    tf_validation_dataset --- TensorFlow Dataset object of the validation dataset.
    ImageEncModel --- Model used to encode the input image.
    HistEncModel --- Model used to encode the input history.
    PathDecModel --- Model used to produce the requested predictions.
    steps_validate --- Number of steps to perform validation on (None means all).
        
    Outputs:
    out_loss --- Mean L2 validation loss.
    '''
    
    idx_val = 0
    print_gen_L2_val = 0
    valLoss_acum = 0 

    valLikelihood_acum = 0
    valTime_displace_acum = 0
        
    val_dataset_prog_bar = tqdm(tf_validation_dataset, total=steps_validate)
    for (valSampleMapComp, valSampeHistPath, valSampeTargetPath, 
             valHistAvail, valTargetAvail, 
         valTimeStamp, valTrackID, valRasterFromAgent) in val_dataset_prog_bar:

        # Predict
        PathDecModel.reset_states()
        HistEncModel.reset_states()
        valPredPath = forwardpass_use(valSampleMapComp,
                                                valSampeHistPath,
                                                valHistAvail,
                                                valSampeTargetPath.shape[-2], 
                                                ImageEncModel, HistEncModel, PathDecModel,
                                                use_teacher_force=False)

        valPredPath = valPredPath.numpy()
        # Calculate loss
        valLoss = calc_loss(valPredPath, valSampeTargetPath, valTargetAvail)
        valLoss = np.mean(valLoss.numpy())

        # Update info
        valLoss_acum += valLoss
        print_gen_L2_val = valLoss_acum/(idx_val+1)


        # Update progress bar
        msg_string = 'Validation: L2 Loss: %.2f (last %.2f) '%(print_gen_L2_val, valLoss)
        val_dataset_prog_bar.set_description(msg_string)

        # # Process lyft metrics
        # for idx_batch in range(valPredPath.shape[0]):
        #     valLikelihood_acum += metrics.neg_multi_log_likelihood(np.expand_dims(valSampeTargetPath[idx_batch,:,:2], axis=0), 
        #                                                             np.expand_dims(valPredPath[idx_batch,:,:2], axis=0), 
        #                                                             np.expand_dims(np.ones((1)), axis=0), 
        #                                                             np.expand_dims(valTargetAvail[idx_batch,:], axis=0), 

        #     valTime_displace_acum += metrics.time_displace(np.expand_dims(valSampeTargetPath[idx_batch,:,:2], axis=0), 
        #                                                             np.expand_dims(valPredPath[idx_batch,:,:2], axis=0), 
        #                                                             np.expand_dims(np.ones((1)), axis=0), 
        #                                                             np.expand_dims(valTargetAvail[idx_batch,:], axis=0), 
                                                                                                                            

        idx_val += 1
        if steps_validate != None:
            if idx_val > steps_validate:
                break
            
    return valLoss_acum/idx_val #, valLikelihood_acum/idx_val, valTime_displace_acum/idx_val



###############################################################################
# ------------------------ LOSS FUNCTIONS ----------------------------------- #
###############################################################################

@tf.function
def calc_loss(predPAth, objPath, availPoints):
    '''
    Calculate the mean L2 loss of the prediction.
    
    Arguments:
    predPAth --- Predicted path.
    objPath --- Real path.
    availPoints --- Number valid real points.   
        
    Outputs:
    out_loss --- Mean L2 loss.
    '''
    
    # Mask with availability
    predPAth = tf.multiply(predPAth, tf.tile(tf.expand_dims(availPoints,axis=-1), [1,1,3]))
    objPath = tf.multiply(objPath, tf.tile(tf.expand_dims(availPoints,axis=-1), [1,1,3]))
    # L2
    out_loss = tf.reduce_sum((predPAth - objPath)**2, axis = -1)
    out_loss = tf.reduce_sum(out_loss, axis = -1)
    
    # Get mean error for each sample in batch
    stepsInfer_byBatch = tf.reduce_sum(availPoints, axis = -1)
    
    return tf.divide(out_loss, stepsInfer_byBatch)
    

    

###############################################################################
# ------------------------ TF DATASET READER -------------------------------- #
###############################################################################

def meta_dict_gen(dataset, 
                  randomize_frame=False, 
                  randomize_scene=False, 
                  num_scenes=16000, 
                  frames_per_scene = -1):
    ''' 
    Dataset generator wrapper for TensorFlow Data compatibility 
    
    Yields a fixed number of scenes and frames
        
    Arguments:
    dataset --- Agent dataset from Lyft library
    randomize_frame --- Frames are yield in random order
    randomize_scene --- Scenes are yield in random order
    num_scenes --- Total number of scenes to be yield in one epoch
    frames_per_scene --- Number of frames to yield per scene (-1 means all)
    '''

    # Scene order (incremental)
    scene_order = np.arange(num_scenes)
    # Shuffle scene order if requested
    if randomize_scene:
        np.random.shuffle(scene_order)

    # For each scene in the dataset
    for scene_idx in scene_order:

        # Frame order (incremental)
        this_scene_idxs = dataset.get_scene_indices(scene_idx)
        # Shuffle frame order if requested
        if randomize_frame:
            np.random.shuffle(this_scene_idxs)
            
        # Get the number of frames to yield from this scene
        if frames_per_scene<1:
            frames_per_scene = this_scene_idxs.shape[0]
        if frames_per_scene > this_scene_idxs.shape[0]:
            frames_per_scene = this_scene_idxs.shape[0]
        # Yield the number of requested frames of this scene
        for frame_idx in range(frames_per_scene):
                yield dataset[this_scene_idxs[frame_idx]]
                
def get_tf_dataset(dataset, 
                   num_hist_frames,
                   map_input_shape,
                   num_future_frames,
                   randomize_frame=False,
                   randomize_scene=False, 
                   num_scenes=16000, 
                   frames_per_scene = -1):
    '''
    Creates a TensorFlow dict dataset from a Lyft dataset generator.
    
    Arguments:
    dataset --- Agent dataset from Lyft library
    num_hist_frames --- Number of history points in a frame
    map_input_shape --- Shape of the map image.
    num_future_frames --- Number of frames to be predicted.
    randomize_frame --- Frames are yield in random order
    randomize_scene --- Scenes are yield in random order
    num_scenes --- Total number of scenes to be yield in one epoch
    frames_per_scene --- Number of frames to yield per scene (-1 means all)
    
    Outputs:
    tf_dataset --- TensorFlow Dataset object.
    '''


    return tf.data.Dataset.from_generator(lambda: meta_dict_gen(dataset,
                                                                randomize_frame=randomize_frame,
                                                                randomize_scene=randomize_scene,
                                                                num_scenes=num_scenes,
                                                                frames_per_scene=frames_per_scene),
                                          output_types={'image': tf.float32, 
                                                        'target_positions': tf.float32,
                                                        'target_yaws': tf.float32,
                                                        'target_availabilities': tf.float32,
                                                        'history_positions': tf.float32,
                                                        'history_yaws': tf.float32,
                                                        'history_availabilities': tf.float32,
                                                        'world_to_image': tf.float32,
                                                        'raster_from_world': tf.float32,
                                                        'raster_from_agent': tf.float32,
                                                        'agent_from_world': tf.float32,
                                                        'world_from_agent': tf.float32,
                                                        'track_id': tf.int32,
                                                        'timestamp': tf.int64,
                                                        'centroid': tf.float32,
                                                        'yaw': tf.float32,
                                                        'extent': tf.float32},
                                          output_shapes={'image': (3+num_hist_frames*2+2, 
                                                                   map_input_shape[0], 
                                                                   map_input_shape[1]), 
                                                         'target_positions': (num_future_frames, 2),
                                                         'target_yaws': (num_future_frames,1),
                                                         'target_availabilities': (num_future_frames,),
                                                         'history_positions': (num_hist_frames+1,2),
                                                         'history_yaws': (num_hist_frames+1,1),
                                                         'history_availabilities': (num_hist_frames+1,),
                                                         'world_to_image': (3,3),
                                                         'raster_from_world': (3,3),
                                                         'raster_from_agent': (3,3),
                                                         'agent_from_world': (3,3),
                                                         'world_from_agent': (3,3),
                                                         'track_id': (),
                                                         'timestamp': (),
                                                         'centroid': (2,),
                                                         'yaw': (),
                                                         'extent': (3,)})


def create_fade_image(input_multiCh_image, fade_factor = 0.75, update_threshold = 0.1):
    '''
    Create an image with previous history layers faded in.
    
    Arguments:
    input_multiCh_image --- Ego or Agent map with channel size equal to number of history frames
    fade_factor --- Intensity fade of the history frames.
    update_threshold --- Minimum image activity to be kept.
        
    Outputs:
    imgFade --- Ego or Agent map with faded history frames.
    '''
    # Get last step image and mask
    imgFade = input_multiCh_image[0,:,:]
    sample_mask = 1.0-tf.cast((imgFade>update_threshold),tf.float32)
    # Process each history point
    for idx_hist in range(1, input_multiCh_image.shape[0]):
        # Get history image and fade it
        auxFadeIn = input_multiCh_image[idx_hist,:,:]*(fade_factor**idx_hist)
        # Apply update mask (do not overwrite recent history)
        auxFadeIn = tf.multiply(sample_mask, auxFadeIn)
        # add to faded image
        imgFade += auxFadeIn
        # Update mask
        sample_mask = 1.0-tf.cast((imgFade>update_threshold),tf.float32)
        
    return imgFade
    

# Sample conformation functions (tf and numpy)
def tf_get_input_sample(datasetSample):
    '''
    Tensorflow sample mapping function.
    
    Takes a frame and pre-process its information, resulting in selected data to 
    be used in the model.
    
    Arguments:
    datasetSample --- Sample tensor (dict).
        
    Outputs:
    sampleMapComp --- Multi-channel image with the RGB map and the faded Ego and Agents
    sampleHistPath --- Tensor of history [num_hist_frames x coordinates]
    sampleTargetPath --- Target tensor [num_objective_frames x coordinates]
    histAvail --- Bool tensor of history availability [num_hist_frames]
    targetAvail --- Bool tensor of target availability [num_objective_frames]
    timeStamp --- Time stamp of the sample
    trackID --- Track ID of the agent
    thisRasterFromAgent --- Raster from Agent conversion matrix.
    '''
    
    # Get number of history frames
    num_hist_frames = datasetSample['history_positions'].shape[0]
    
    # Map to RGB
    sampleMap = tf.transpose(datasetSample['image'][-3:,:,:], perm=[1, 2, 0])
    sampleMap *= 255
    
    # Ego to fade
    sampleEgoFade = create_fade_image(datasetSample['image'][-3-num_hist_frames-1:-3,:,:])
    sampleEgoFade /= tf.reduce_max(sampleEgoFade)
    sampleEgoFade *= 255
    sampleEgoFade = tf.expand_dims(sampleEgoFade, axis=-1)
    
    
    
    # Agents to fade
    sampleAgentsFade = create_fade_image(datasetSample['image'][:-3-num_hist_frames-1,:,:])
    sampleAgentsFade /= (tf.reduce_max(sampleAgentsFade)+1e-8) # Add a low value in case no other agent is present
    sampleAgentsFade *= 255
    sampleAgentsFade = tf.expand_dims(sampleAgentsFade, axis=-1)
    
    # Concatenate image inputs
    sampleMapComp = tf.concat([sampleMap, sampleEgoFade, sampleAgentsFade], axis = -1)
    
    # History path
    x = datasetSample['history_positions'][:,0]#/100.0
    x = tf.expand_dims(x, axis=-1)
    y = datasetSample['history_positions'][:,1]#/40.0
    y = tf.expand_dims(y, axis=-1)
    a = datasetSample['history_yaws']#/np.pi
    sampleHistPath = tf.concat([x, y, a], axis = -1)   
    
    
    # Targets
    x = datasetSample['target_positions'][:,0]#/100.0
    x = tf.expand_dims(x, axis=-1)
    y = datasetSample['target_positions'][:,1]#/40.0
    y = tf.expand_dims(y, axis=-1)
    a = datasetSample['target_yaws']#/np.pi
    sampleTargetPath = tf.concat([x, y, a], axis = -1)  
    
    # Availability
    histAvail = datasetSample['history_availabilities']
    targetAvail = datasetSample['target_availabilities']
#     histAvail = datasetSample['history_availabilities']
#     targetAvail = datasetSample['target_yaws']!=0.0
#     targetAvail = tf.squeeze(targetAvail)
    # targetAvail_mask = tf.cast(tf.concat([[1],tf.zeros(targetAvail.shape[0]-1)], axis=0), dtype=tf.bool) # Force at least one sample to true (this fails, and I dont trust the dataset availability...)
    # targetAvail = tf.cast(tf.logical_or(targetAvail, targetAvail_mask), dtype=np.float32)

    # Data
    timeStamp = datasetSample['timestamp']
    trackID = datasetSample['track_id']
    thisRasterFromAgent = datasetSample["raster_from_agent"]

    return sampleMapComp, sampleHistPath, sampleTargetPath, histAvail, targetAvail, timeStamp, trackID, thisRasterFromAgent

    
    
    
###############################################################################
# ------------------------ TF MODEL LOAD/SAVE ------------------------------- #
###############################################################################


def save_model(model_save, save_path, save_name, use_keras=True):
    """
    Save model using Keras API (single .h5 file) OR a HDF5 for weights and JSON for model.
    
    Arguments:
    model_save --- Keras model to save
    save_path --- Output folder
    save_name --- Name of the h5 file to be written
    use_keras --- Use keras API to save, else use JSON + HDF5
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if use_keras:
        model_save.save(os.path.join(save_path,save_name+".h5"))

    else:
        # serialize model to JSON
        model_json = model_save.to_json()
        with open(os.path.join(save_path,save_name+".json"), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_save.save_weights(os.path.join(save_path,save_name+".h5"))
    
    return

def load_model(load_path, load_name, use_keras=True, custom_obj_dict=[]):
    """
    Load model using Keras API (single .h5 file) OR from a HDF5 file for weights and a JSON file for model  
    
    Arguments:
    load_path --- Model save folder
    load_name --- Name of the h5 file to be read
    use_keras --- Use keras API to save, else use JSON + HDF5
    custom_obj_dict --- Dictionary of custom objects for keras API (i.e. custom layers)
    
    Outputs:
    model --- TensorFlow Keras model.
    """


    if use_keras:
        loaded_model = keras.models.load_model(os.path.join(load_path,load_name+".h5"), custom_objects=custom_obj_dict)

    else:
        MODEL_NAME = load_name+'.json'
        WEIGHTS_NAME = load_name+'.h5'
        # load json and create model
        json_file = open(os.path.join(load_path,load_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(load_path,load_name))
    
    return loaded_model