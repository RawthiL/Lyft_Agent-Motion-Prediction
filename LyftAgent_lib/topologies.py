#!/usr/bin/python
# -*- coding: iso-8859-15 -*-


import os, sys
import numpy as np

from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K



###############################################################################
# ------------------------ COMPOUND MODEL FORWARD PASS ---------------------- #
###############################################################################

@tf.function
def modelV2_forward_pass(img_t, hist_t, histAvail, stepsInfer, ImageEncModel, HistEncModel, PathDecModel,
                        thisHiddenState = None,
                       use_teacher_force = False, 
                       teacher_force_weight = tf.constant(1.0), 
                       target_path=None,
                       stop_gradient_on_prediction = False,
                       gradient_clip_thorugh_time_value = 10.0,
                       return_attention = False,
                       training_state=False):
    '''
    Forward pass of the path decoding model V2.
    
    This function applies all models to get features and the number of requested
    predictions.

    Unlike model V1, this model uses the HistEncModel only for coding the history and then
    is not updated with predicted positions. The output sequence of the HistEncModel is
    used with the attention mechanism of the PathDecMode. 
    Predicted positions are re-used by the PathDecModel concatenated to the image and history
    feature tensors. 
    The initial state of the PathDecModel is obtained from the last state of the HistEncModel.
    
    Arguments:
    img_t --- Multi-channel map input tensor.
    hist_t --- Path history tensor.
    histAvail --- History availability tensor.
    stepsInfer --- Number of steps to infer.
    ImageEncModel --- Model used to encode the input image (img_t)
    HistEncModel --- Model used to encode the input history (hist_t)
    PathDecModel --- Model used to produce the requested predictions.
    thisHiddenState --- Not used, only for compat.
    use_teacher_force --- Use a linear interpolation of the target path and the last predicted path.
    teacher_force_weight --- Fraction of the target path used in the linear interpolation between target and predicted.
    target_path --- Target path (used if use_teacher_force==True)
    stop_gradient_on_prediction --- Prevent gradient to be calculated through the recursive net.
    
        
    Outputs:
    out_list --- Predicted path points.
    '''
                       
    # Reset the hidden states for this batch # BUGGED INSIDE @tf.function??
    HistEncModel.reset_states()
    
    # Number of history states
    stepsHist = hist_t.shape[1]
    #stepsHist = tf.reduce_sum(histAvail, axis=-1)
    

    # Process input image
    img_feats = ImageEncModel(img_t, training = training_state)  

    # Process history
    hist_out_list = list()
    hist_hidden_list = list()
    for idx_step_hist in range(stepsHist):
        hist_outs, hist_states = HistEncModel(tf.expand_dims(hist_t[:,stepsHist-1-idx_step_hist,:], 1), training = training_state)

        # Clip gradients each time step!!
        hist_outs = clip_gradients_000001(hist_outs)
        hist_states = clip_gradients_000001(hist_states)

        hist_out_list.append(hist_outs)
        hist_hidden_list.append(hist_states)

    # Convert results list to tensor
    hist_outs = tf.convert_to_tensor(hist_out_list)
    hist_states = tf.convert_to_tensor(hist_hidden_list)
    # Transpose match dimension order
    hist_outs = tf.transpose(hist_outs, (1,0,2))
    hist_states = tf.squeeze(hist_states)
    hist_states = tf.transpose(hist_states, (1,0,2))
        
    # Reset decoder model states to last state of HistEncModel
    thisHiddenState = tf.squeeze(hist_states[:,-1,:])
    reset_states(PathDecModel, layer_states_dict = {'gru':  thisHiddenState})
                       
    # Output List
    out_list = list()
    if return_attention:
        out_img_attention = list()
        out_hist_attention = list()

    # Create requested path points
    nextPath = tf.squeeze(hist_t[:,0,:]) # Process current step first
    for idx_step in range(stepsInfer):
        
        if stop_gradient_on_prediction:
            nextPath = tf.stop_gradient(nextPath)

        # Clip gradients each time step!!
        nextPath = clip_gradients_000001(nextPath)
        img_feats = clip_gradients_000001(img_feats)
        hist_outs = clip_gradients_000001(hist_outs)
        hist_states = clip_gradients_000001(hist_states)
        thisHiddenState = clip_gradients_000001(thisHiddenState)
            
        # Get predicted step using decoder
        thisPath, thisHiddenState, atten_img, atten_hist = PathDecModel([nextPath,
                                                       img_feats, 
                                                       hist_outs, 
                                                       hist_states, 
                                                       thisHiddenState], training = training_state)
        
        # Add out step
        out_list.append(thisPath)
        if return_attention:
            out_img_attention.append(atten_img)
            out_hist_attention.append(atten_hist)

        # Apply teacher forcing
        if use_teacher_force:
            # Our teacher force can be modulated using linear interpolation between the 
            # real and generated path
            delta_path = (thisPath-target_path[:,idx_step,:])
            nextPath = target_path[:,idx_step,:] + (1.0-teacher_force_weight)*delta_path
            
        else:
            nextPath = thisPath
        
        
        
    # Convert results list to tensor
    out_list = tf.convert_to_tensor(out_list)
    # Transpose match dimension order
    out_list = tf.transpose(out_list, (1,0,2))
    
    if return_attention:
        return out_list, out_img_attention, out_hist_attention
    else:
        return out_list


@tf.function
def modelV1_forward_pass(img_t, hist_t, histAvail, stepsInfer, ImageEncModel, HistEncModel, PathDecModel,
                       thisHiddenState = None,
                       use_teacher_force = False, 
                       teacher_force_weight = tf.constant(1.0), 
                       target_path=None,
                       stop_gradient_on_prediction = False):
    '''
    Forward pass of the path decoding model V1.
    
    This function applies all models to get features and the number of requested
    predictions.
    
    Arguments:
    img_t --- Multi-channel map input tensor.
    hist_t --- Path history tensor.
    histAvail --- History availability tensor.
    stepsInfer --- Number of steps to infer.
    ImageEncModel --- Model used to encode the input image (img_t)
    HistEncModel --- Model used to encode the input history (hist_t)
    PathDecModel --- Model used to produce the requested predictions.
    thisHiddenState --- Initial hidden state of the decoder model.
    use_teacher_force --- Use a linear interpolation of the target path and the last predicted path.
    teacher_force_weight --- Fraction of the target path used in the linear interpolation between target and predicted.
    target_path --- Target path (used if use_teacher_force==True)
    stop_gradient_on_prediction --- Prevent gradient to be calculated through the recursive net.
    
        
    Outputs:
    out_list --- Predicted path points.
    '''
                       
    # Reset the hidden states for this batch # BUGGED INSIDE @tf.function??
#     PathDecModel.reset_states()
#     HistEncModel.reset_states()
    
    # Number of history states
    stepsHist = hist_t.shape[1]
    #stepsHist = tf.reduce_sum(histAvail, axis=-1)
    
    if not tf.is_tensor(thisHiddenState):
        thisHiddenState = tf.zeros(PathDecModel.inputs[-1].shape)
    
    # Process input image
    img_feats = ImageEncModel(img_t)  

    # Process history
    for idx_step_hist in range(stepsHist-1):
        hist_outs, hist_feats = HistEncModel(tf.expand_dims(hist_t[:,stepsHist-1-idx_step_hist,:], 1))
        
        # Get predicted path using decoder (to have a warm start of the hidden state)
        thisPath, thisHiddenState = PathDecModel([img_feats, hist_outs, hist_feats, thisHiddenState])
                       
    # Output List
    out_list = list()

    # Create requested path points
    nextPath = hist_t[:,0,:] # Process current step first
    for idx_step in range(stepsInfer):
        
        # Update history encoding
        nextPath = tf.expand_dims(nextPath, 1)
        if stop_gradient_on_prediction:
            nextPath = tf.stop_gradient(nextPath)
        hist_outs, hist_feats = HistEncModel(nextPath)

        # Get predicted step using decoder
        thisPath, thisHiddenState = PathDecModel([img_feats, hist_outs, hist_feats, thisHiddenState])
        
        # Add out step
        out_list.append(thisPath)

        # Apply teacher forcing
        if use_teacher_force:
            # Our teacher force can be modulated using linear interpolation between the 
            # real and generated path
            delta_path = (thisPath-target_path[:,idx_step,:])
            nextPath = target_path[:,idx_step,:] + (1.0-teacher_force_weight)*delta_path
            
        else:
            nextPath = thisPath
        
        
        
    # Convert results list to tensor
    out_list = tf.convert_to_tensor(out_list)
    # Transpose match dimension order
    out_list = tf.transpose(out_list, (1,0,2))
    
    return out_list




###############################################################################
# ------------------------ PATH PREDICTION DECODING NETS -------------------- #
###############################################################################


def pathDecoderModel_V2(cfg, ImageEncModel, HistEncModel):
    '''
    Creates a Keras model for Agent path prediction based on a given recurrent unit.
    
    This model takes as input the code tensors from the encoding models and outputs 
    a predicted path. This models implements attention on the image code tensor and
    on the states of a previous recurrent unit used to encode the history path.
        
    Arguments:
    cfg --- Configuration.
    ImageEncModel --- Keras image encoding model.
    HistEncModel --- Keras image encoding model.
    
        
    Outputs:
    PathDecModel --- Keras path decoding model.
    '''

    # Get config
    gen_batch_size             = cfg["train_data_loader"]["batch_size"]
    histEnc_recurrent_unit_num = cfg["model_params"]["history_encoder_recurrent_units_number"]
    pathDec_recurrent_unit     = cfg["model_params"]["path_generation_decoder_recurrent_unit"]
    pathDec_recurrent_unit_num = cfg["model_params"]["path_generation_decoder_recurrent_units_number"]
    pathDec_attention_pr_units = cfg["model_params"]["pathDec_attention_pr_units"]
    num_path_decode_fc_layers  = cfg["model_params"]["num_path_decode_fc_layers"]
    path_decode_fc_units       = cfg["model_params"]["path_decode_fc_units"]
    path_decode_fc_activation  = cfg["model_params"]["path_decode_fc_activation"]
    increment_net              = cfg["model_params"]["increment_net"]
    path_noise_level           = cfg["model_params"]["path_noise_level"]

    imageProcPixelNum          = ImageEncModel.output_shape[1]
    imageProcFeatNum           = ImageEncModel.output_shape[2]
    
    # Inputs
    Input_Image_Features = keras.Input(batch_shape=(gen_batch_size, imageProcPixelNum, imageProcFeatNum), 
                                       name="Input_Image_Features")
    Input_History_Hidden = keras.Input(batch_shape=(gen_batch_size, None, histEnc_recurrent_unit_num), 
                                         name="Input_History_Hidden")
    Input_History_Out = keras.Input(batch_shape=(gen_batch_size, None, histEnc_recurrent_unit_num), 
                                         name="Input_History_Out")
    Input_Last_Hidden_State = keras.Input(batch_shape=(gen_batch_size, pathDec_recurrent_unit_num), 
                                         name="Input_Last_Hidden_State")
    Input_position = keras.Input(batch_shape=(gen_batch_size, 3), name="Input_position")

    # Add noise
    in_position = tf.keras.layers.GaussianNoise(path_noise_level)(Input_position)

    # Set attention mechanism for image
    imageAttnFeatT, attenScores_image = BahdanauAttention(pathDec_attention_pr_units)([Input_Last_Hidden_State, Input_Image_Features])
    # Set attention mechanism for history
    histAttnFeatT, attenScores_history = BahdanauAttention(pathDec_attention_pr_units)([Input_Last_Hidden_State, Input_History_Out])


    # Project the physical attention to a lower space (matching path history encoding)
    imageAttnFeatT = keras.layers.Dense(histEnc_recurrent_unit_num,
                                        kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                        activation=keras.layers.LeakyReLU(alpha=0.1))(imageAttnFeatT)

    # Concatenate feature tensors with current position
    featT = keras.layers.Concatenate(axis=-1)([imageAttnFeatT, histAttnFeatT, in_position])


    # Get recurrent unit to use
    try:
        pathDec_base_recurrent_unit = getattr(keras.layers, pathDec_recurrent_unit)
    except:
        raise Exception('Path decoder base recurrent unit not found. Requested unit: %s'%pathDec_recurrent_unit)

    # Apply recurrent output
    featT = K.expand_dims(featT, 1)
    featOutT, featHidT  = pathDec_base_recurrent_unit(pathDec_recurrent_unit_num,
                                                        kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                                        recurrent_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                                        # bias_regularizer=keras.regularizers.l2(0.001),
                                                      stateful=True,
                                                      return_state=True)(featT)

    # Get predicted position
    for idx_decode_layer in range(num_path_decode_fc_layers):
        featOutT = keras.layers.Dense(path_decode_fc_units[idx_decode_layer],
                                      kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                      activation = path_decode_fc_activation)(featOutT)
    # Last output
    pathOut = keras.layers.Dense(3,
                                kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                activation = None)(featOutT)

    if increment_net:
        pathOut += Input_position

    # Create model
    return keras.Model([Input_position,
                        Input_Image_Features,
                        Input_History_Out, 
                        Input_History_Hidden, 
                        Input_Last_Hidden_State],
                       [pathOut, featHidT, attenScores_image, attenScores_history],
                       name='Path_Decoder_Model')


def pathDecoderModel_V1(cfg, ImageEncModel, HistEncModel):
    '''
    Creates a Keras model for Agent path prediction based on a given recurrent unit.
    
    This model takes as input the code tensors from the encoding models and outputs 
    a predicted path. This models implements attention on the image code tensor.
    
    Arguments:
    cfg --- Configuration.
    ImageEncModel --- Keras image encoding model.
    HistEncModel --- Keras image encoding model.
    
        
    Outputs:
    PathDecModel --- Keras path decoding model.
    '''

    # Get config
    gen_batch_size             = cfg["train_data_loader"]["batch_size"]
    histEnc_recurrent_unit_num = cfg["model_params"]["history_encoder_recurrent_units_number"]
    pathDec_recurrent_unit     = cfg["model_params"]["path_generation_decoder_recurrent_unit"]
    pathDec_recurrent_unit_num = cfg["model_params"]["path_generation_decoder_recurrent_units_number"]
    pathDec_attention_pr_units = cfg["model_params"]["pathDec_attention_pr_units"]
    num_path_decode_fc_layers  = cfg["model_params"]["num_path_decode_fc_layers"]
    path_decode_fc_units       = cfg["model_params"]["path_decode_fc_units"]
    path_decode_fc_activation  = cfg["model_params"]["path_decode_fc_activation"]

    imageProcPixelNum          = ImageEncModel.output_shape[1]
    imageProcFeatNum           = ImageEncModel.output_shape[2]
    
    # Inputs
    Input_Image_Features = keras.Input(batch_shape=(gen_batch_size, imageProcPixelNum, imageProcFeatNum), 
                                       name="Input_Image_Features")
    Input_History_Features = keras.Input(batch_shape=(gen_batch_size, histEnc_recurrent_unit_num), 
                                         name="Input_History_Features")
    Input_History_Out = keras.Input(batch_shape=(gen_batch_size, histEnc_recurrent_unit_num), 
                                         name="Input_History_Out")
    Input_Last_Hidden_State = keras.Input(batch_shape=(gen_batch_size, pathDec_recurrent_unit_num), 
                                         name="Input_Last_Hidden_State")


    histAttenFeatT = Input_History_Features
    # Set attention mechanism for image
    # Image_query = keras.layers.Concatenate(axis=-1)([Input_History_Features, Input_Last_Hidden_State])
    Image_query = Input_Last_Hidden_State

    # img_att_q = keras.layers.Dense(imageProcFeatNum)(Input_Last_Hidden_State)
    # img_att_q = K.reshape(img_att_q, shape=[-1, 1, img_att_q.shape[1]])
    # imageAttenFeatT = tf.keras.layers.AdditiveAttention(causal = False)([img_att_q, Input_Image_Features])
    # imageAttenFeatT = K.reshape(imageAttenFeatT, shape=[-1, imageProcFeatNum])

    # imageAttenFeatT, attenScores = BahdanauAttention(pathDec_attention_pr_units)([Input_Last_Hidden_State, Input_Image_Features])
    imageAttenFeatT, attenScores = BahdanauAttention(pathDec_attention_pr_units)([Image_query, Input_Image_Features])



    # Project the physical attention to a lower space (matching path history encoding)
    imageAttenFeatT = keras.layers.Dense(histEnc_recurrent_unit_num,
                                         activation=keras.layers.LeakyReLU(alpha=0.1))(imageAttenFeatT)

    # Concatenate feature tensors
    # featT = keras.layers.Concatenate(axis=-1)([imageAttenFeatT, histAttenFeatT])
    featT = keras.layers.Concatenate(axis=-1)([imageAttenFeatT, Input_History_Out])


    # Get recurrent unit to use
    try:
        pathDec_base_recurrent_unit = getattr(keras.layers, pathDec_recurrent_unit)
    except:
        raise Exception('Path decoder base recurrent unit not found. Requested unit: %s'%pathDec_recurrent_unit)

    # Apply recurrent output
    featT = K.expand_dims(featT, 1)
    featOutT, featHidT  = pathDec_base_recurrent_unit(pathDec_recurrent_unit_num,
                                                      stateful=True,
                                                      return_state=True)(featT)

    # Get predicted position
    for idx_decode_layer in range(num_path_decode_fc_layers):
        featOutT = keras.layers.Dense(path_decode_fc_units[idx_decode_layer],
                                      activation = path_decode_fc_activation)(featOutT)
    # Last output
    pathOut = keras.layers.Dense(3,
                                activation = None)(featOutT)

    # Create model
    return keras.Model([Input_Image_Features,
                        Input_History_Out, 
                        Input_History_Features, 
                        Input_Last_Hidden_State],
                       [pathOut, featHidT],
                       name='Path_Decoder_Model')





###############################################################################
# ------------------------ PATH HISTORY ENCODING NET ------------------------ #
###############################################################################

def pathEncodingModel(cfg, return_sequences=False, CarNet = False):
    '''
    Creates a Keras model for Agent history encoding based on a given recurrent unit.
    
    Arguments:
    cfg --- Configuration.
    return_sequences --- If true the outputs and states for each input step are returned
    CarNet --- If true, only the dense layers are applied, making the V1 model the same as the CARnet (sadeghian2018car)
        
    Outputs:
    HistEncModel --- Keras path history encoding model.
    '''

    # Get config
    gen_batch_size             = cfg["train_data_loader"]["batch_size"]
    histEnc_recurrent_unit     = cfg["model_params"]["history_encoder_recurrent_unit"]
    histEnc_recurrent_unit_num = cfg["model_params"]["history_encoder_recurrent_units_number"]
    num_hist_encode_layers     = cfg["model_params"]["num_hist_encode_layers"]
    hist_encode_units          = cfg["model_params"]["hist_encode_units"]
    path_noise_level           = cfg["model_params"]["path_noise_level"]
    # hist_encode_activation     = cfg["model_params"]["hist_encode_activation"]
    hist_encode_activation     = keras.layers.LeakyReLU(alpha=0.1)

    # Inputs
    Input_hist_frames = keras.Input(batch_shape=(gen_batch_size, 1, 3), 
                                    name="Input_history_frames")

    # Add noise
    pathFeatT = tf.keras.layers.GaussianNoise(path_noise_level)(Input_hist_frames)

    # History encoding dense layers
    for idx_encode_layer in range(num_hist_encode_layers):
        pathFeatT = keras.layers.Dense(hist_encode_units[idx_encode_layer],
                                       kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                       activation=hist_encode_activation)(pathFeatT)

    if not CarNet:
        # Get recurrent unit to use
        try:
            histEnc_base_recurrent_unit = getattr(keras.layers, histEnc_recurrent_unit)
        except:
            raise Exception('History encoder base recurrent unit not found. Requested unit: %s'%histEnc_recurrent_unit)
    
        # Recurrent encoding
        outs  = histEnc_base_recurrent_unit(histEnc_recurrent_unit_num,
                                            kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                            recurrent_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                            # bias_regularizer=keras.regularizers.l2(0.001),
                                            return_state=True,
                                            return_sequences=return_sequences,
                                            stateful=True)(pathFeatT)
        # Get output and states
        encoder_out = outs[0]
        encoder_states = outs[1:]
    else:
        # Return the encoding of the input coordinates
        encoder_out = pathFeatT
        encoder_states = pathFeatT

    

    return keras.Model(Input_hist_frames, [encoder_out, encoder_states], name='History_Encoding_Model')


###############################################################################
# ------------------------ IMAGE ENCODING ----------------------------------- #
###############################################################################

def imageEncodingModel(base_img_model, cfg):
    '''
    Creates a Keras model for image encoding based on a given convolutional network.
    
    Arguments:
    base_img_model --- Keras base image encoding model.
    cfg --- Configuration.
    
    
    Outputs:
    ImageEncModel --- Keras image encoding model.
    '''
    
    # Get config
    model_map_input_shape      = (cfg["raster_params"]["raster_size"][0],
                                  cfg["raster_params"]["raster_size"][1])
    retrain_inputs_image_model = cfg["training_params"]["retrain_inputs_image_model"]
    retrain_all_image_model = cfg["training_params"]["retrain_all_image_model"]
    retrain_inputs_image_model = retrain_all_image_model or retrain_inputs_image_model
    map_input_channels         = 3+2 # RGB + Ego Fade + Agent Fade
   
    # Inputs
    Input_map = keras.Input(shape=(model_map_input_shape[0],
                                       model_map_input_shape[1], 
                                       map_input_channels), 
                                name="Input_map")
    mapT = Input_map

    # Redefine input layer of base image model (for multiple channel compatibility)
    # First, get first convolutional layer position
    idx_layer_start = 0
    layer_start = base_img_model.layers[idx_layer_start]
    while not isinstance(layer_start, keras.layers.Conv2D):
        idx_layer_start += 1
        layer_start = base_img_model.layers[idx_layer_start]

    # Create a new layer with the same characteristics
    newConv = keras.layers.Conv2D(layer_start.filters,
                                  layer_start.kernel_size,
                                  strides=layer_start.strides,
                                  padding=layer_start.padding,
                                  data_format=layer_start.data_format,
                                  dilation_rate=layer_start.dilation_rate,
                                  activation=layer_start.activation,
                                  use_bias=layer_start.use_bias,
                                  kernel_initializer=layer_start.kernel_initializer,
                                  bias_initializer=layer_start.bias_initializer,
                                  kernel_regularizer=layer_start.kernel_regularizer,
                                  bias_regularizer=layer_start.bias_regularizer,
                                  activity_regularizer=layer_start.activity_regularizer,
                                  kernel_constraint=layer_start.kernel_constraint,
                                  bias_constraint=layer_start.bias_constraint,
                                  trainable = retrain_inputs_image_model)

    mapT = newConv(mapT)

    # Set RGB filter channels to the pretrained values
    # For the new channels we will copy the red channel 
    # (most important for human vision, maybe also in this net?, nevertheless we will retrain them...)
    this_weights = layer_start.get_weights()
    new_weights = np.zeros((layer_start.kernel_size[0], layer_start.kernel_size[1], map_input_channels, layer_start.filters))
    new_weights[:,:,:3,:] = this_weights[0]
    for idx_newChn in range(map_input_channels-this_weights[0].shape[2]):
        new_weights[:,:,this_weights[0].shape[2]+idx_newChn,:] = this_weights[0][:,:,0,:]

    newConv.set_weights([new_weights])



    # Add the rest of the base image processing model, loading its weights...
    # Since the model is not sequential, this will be a complex process.
    # For each layer added we need to track its name and save the node for future connections.
    bim_layers = dict()
    bim_layers[layer_start.name] = mapT
    for idx_layer, this_layer in enumerate(base_img_model.layers):

        # Before input convolution
        if idx_layer <= idx_layer_start:
            continue 

        # Get name of predecessor layers
        int_node = this_layer._inbound_nodes[0]
        if type(int_node.inbound_layers) is list:
            prev_layers_names = [auxLayer.name for auxLayer in int_node.inbound_layers]
            layer_input_list = [bim_layers.get(key) for key in prev_layers_names]
        else:
            prev_layers_names = int_node.inbound_layers.name
            layer_input_list = bim_layers.get(prev_layers_names)

        # Get layer configuration
        this_config = this_layer.get_config()
        this_weights = this_layer.get_weights()
        # Freeze the layer weights
        this_config['trainable'] = retrain_all_image_model
        # Duplicate layer
        append_layer = this_layer.from_config(this_config)
        # Connect to all inbound layers   
        mapT = append_layer(layer_input_list)
        # Reload pretrained weights
        append_layer.set_weights(this_weights)




        # Save layer info
        bim_layers[this_layer.name] = mapT


    # Flatten to feature tensor 
    # mapFeatT = keras.layers.Flatten()(mapT)
    # Reshape feature map
    mapFeatT = K.reshape(mapT, shape=[-1,mapT.shape[1]*mapT.shape[2],mapT.shape[3]])

    return keras.Model(Input_map, mapFeatT, name='Image_Encodding_Model')
    
    


###############################################################################
# ------------------------ CUSTOM LAYERS AND FUNCTIONS ---------------------- #
###############################################################################

# def clip_gradients(y):
#     return tf.py_function(func=tf_clip_gradients, inp=[y], Tout=tf.float32)


# Custom gradient clip for RNN
@tf.custom_gradient
def clip_gradients_00001(y):
    def backward(dy):
        return tf.clip_by_norm(dy, 0.1)
        # return tf.clip_by_value(dy, clip_value_max=0.01, clip_value_min=-0.01)
    return y, backward

@tf.custom_gradient
def clip_gradients_000001(y):
    def backward(dy):
        return tf.clip_by_norm(dy, 0.1)
        # return tf.clip_by_value(dy, clip_value_max=0.001, clip_value_min=-0.001)
    return y, backward


# Attention layer from https://www.tensorflow.org/tutorials/text/nmt_with_attention
# (with some modifications for save/load operations)
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(self.units)
        self.W2 = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x):
        
        query, values = x
        
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config


    
def reset_states(model, layer_states_dict = dict()):
    '''
    Resets the state of the "statefull" recurrent layers within a model.
    Optionally sets the states to a given value.

    Arguments:
    model --- Model to be resetted.
    layer_states_dict --- Dictionary containing the name and value of each recurrent layer to be initialized.
    '''
    for layer in model.layers:
        if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):

            if layer.name in layer_states_dict:
                layer.reset_states(states = layer_states_dict[layer.name])
            else:
                layer.reset_states()

    return