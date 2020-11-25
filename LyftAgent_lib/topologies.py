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

def compute_mruv_path(vel, accel, stepsInfer, clip_low_lims = [0.0, -80.0], clip_high_lims = [200.0, 80.0]):
    """
    Computes the futures path using uniform variate linear movement:
    x = xo + s*t + a*t^2

    Args:
        vel (np.array): Velocity of each agent in batch.
        accel (np.array): Acceleration of each agent in batch.
        stepsInfer (np.int32): Number of steps to calculate.
        clip_low_lims (list): Minimum allowed position for X and Y cord (agent view).
        clip_high_lims (list): Maximum allowed position for X and Y cord (agent view).
    Returns:
        mruv_path (tensor): path for each agent in the batch.
    """

    mruv_path = list()
    for idx_step in range(0, stepsInfer+1):
        # Get position, considering t = idx_step
        this_mruv_pos = (idx_step*vel) + (accel*idx_step*idx_step/2.0)
        this_mruv_pos = tf.clip_by_value(this_mruv_pos, clip_low_lims, clip_high_lims)
        mruv_path.append(this_mruv_pos) 
    mruv_path = tf.convert_to_tensor(mruv_path)
    mruv_path = tf.transpose(mruv_path, (1,0,2))

    return mruv_path


@tf.function
def modelV2_forward_pass(img_t, hist_t, histAvail, stepsInfer, ImageEncModel, HistEncModel, PathDecModel,
                       use_teacher_force = False, 
                       teacher_force_weight = tf.constant(1.0), 
                       target_path=None,
                       stop_gradient_on_prediction = False,
                       gradient_clip_thorugh_time_value = 10.0,
                       return_attention = False,
                       training_state = False,
                       increment_net = False,
                       mruv_guiding = False,
                       mruv_model = None,
                       mruv_model_trainable = False, **kwargs):
    """
    Forward pass of the path decoding model V2.
    
    This function applies all models to get features and the number of requested
    predictions.

    Unlike model V1, this model uses the HistEncModel only for coding the history and then
    is not updated with predicted positions. The output sequence of the HistEncModel is
    used with the attention mechanism of the PathDecMode (if selected, otherwise it makes a mean over all features). 
    Predicted positions are re-used by the PathDecModel concatenated to the image and history
    feature tensors (and mruv guiding if selected). 
    The initial state of the PathDecModel is obtained from the last state of the HistEncModel (this is bugged in graph mode...).
    
    Args:
        img_t (tf.tensor / np.array): Tensor with input map batch, multi channel image shape: [batch, x, y, ch].
        hist_t (tf.tensor / np.array): Path history tensor with shape: [batch, hist_steps, coords].
        HistAvail  (tf.tensor / np.array): Tensor with history path availability (boolean), shape: [batch, steps, coords].
        stepsInfer (np.int): Number of future steps to predict.
        ImageEncModel (keras.model): Keras model for image encoding.
        HistEncModel  (keras.model): Keras model for history path encoding.
        PathDecModel (keras.model): Keras model for path decoding/generation.
        use_teacher_force (np.bool): Whether or not use teacher force during training.
        teacher_force_weight (np.float): How much does the teacher force modifies the RNN output.
        target_path (tf.tensor / np.array): Tensor with objective path, shape: [batch, steps, coords].
        stop_gradient_on_prediction (np.bool): Whether or not stop the gradient from flowing back through the RNN steps.
        gradient_clip_thorugh_time_value (np.float): Value of the gradient clipping during the backward pass, not used because it is not supported in a tf.function, the clip value must be fixed.
        return_attention (np.bool): Return attention weights.
        training_state (np.bool): True -> Training, False -> Inference.
        increment_net (np.bool): If this is an increment net the RNN only predicts the step length, not the absolute positions.
        mruv_guiding (np.bool): If true a model is used to calculate the speed and acceleration of the vehicle and infer the future steps using accelerated linear movement. This is fed to the network as an additional input.
        mruv_model (keras.model / function): A model or a function which generates the speed and acceleration for the batch.
        mruv_model_trainable (np.boolean): If the mruv_model is trainable we must get gradients out of it.
     
    Returns:
        out_list (tf.tensor): Predicted path points.
        out_img_attention (tf.tensor): Attention weights for the image features (if return_attention == True)
        out_hist_attention (tf.tensor): Attention weights for the history path features (if return_attention == True)
        mruv_v (tf.tensor): speed value of each agent in the batch. (if mruv_model == True and mruv_model_trainable == True)
        mruv_a (tf.tensor): acceleration value of each agent in the batch. (if mruv_model == True and mruv_model_trainable == True)
        mruv_confidence (tf.tensor): confidence value of the speed and acceleration of each agent in the batch. (if mruv_model == True and mruv_model_trainable == True)
    """
                       
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
        hist_outs = clip_gradients_01(hist_outs)
        hist_states = clip_gradients_01(hist_states)

        hist_out_list.append(hist_outs)
        hist_hidden_list.append(hist_states)

    # Convert results list to tensor
    hist_outs = tf.convert_to_tensor(hist_out_list)
    hist_states = tf.convert_to_tensor(hist_hidden_list)
    # Transpose match dimension order
    hist_outs = tf.transpose(hist_outs, (1,0,2))
    hist_states = tf.squeeze(hist_states)
    hist_states = tf.transpose(hist_states, (1,0,2))

    # Get uniform movement data
    if mruv_guiding:
        if mruv_model_trainable:
            mruv_v, mruv_a, mruv_confidence = mruv_model([hist_outs[:,-1,:], img_feats], training = training_state)
        else:
            mruv_v, mruv_a, mruv_confidence = mruv_model(hist_t)
        # Compute future steps
        mruv_path = compute_mruv_path(mruv_v, mruv_a, stepsInfer)
        
    # Reset decoder model states to last state of HistEncModel # NOT WORKING; WHY????
    thisHiddenState = tf.squeeze(hist_states[:,-1,:])
    # reset_states(PathDecModel, layer_states_dict = {'pathRecUnit':  thisHiddenState}) # Bugged in graph mode....
    PathDecModel.reset_states() # Reset to zero ...

                     
    # Output List
    out_list = list()
    if return_attention:
        out_img_attention = list()
        out_hist_attention = list()

    # Create requested path points
    nextPath = tf.squeeze(hist_t[:,0,:]) # Process current step first
    nextPath = clip_gradients_01(nextPath)
    prev_path = hist_t[:,1,:]
    for idx_step in range(stepsInfer):
        
        # Stop gradient if requested
        if stop_gradient_on_prediction:
            nextPath = tf.stop_gradient(nextPath)
            thisHiddenState = tf.stop_gradient(thisHiddenState)

        # Clip gradients each time step!!
        nextPath = clip_gradients_01(nextPath)
        img_feats = clip_gradients_01(img_feats)
        hist_outs = clip_gradients_01(hist_outs)
        hist_states = clip_gradients_01(hist_states)
        thisHiddenState = clip_gradients_01(thisHiddenState)

        # If this is an increment net, provide only the step size
        if increment_net:
            inPath = nextPath-prev_path
        else: 
            inPath = nextPath

        # Create input list of the decoder RNN
        input_list = [inPath, img_feats, hist_outs, hist_states, thisHiddenState]
        # If ruv guiding is used, add the path/step calculated using linear movement
        if mruv_guiding:
            if increment_net:
                mruvInPath = mruv_path[:,idx_step+1,:]-mruv_path[:,idx_step,:]
            else: 
                mruvInPath = mruv_path[:,idx_step+1,:]
            input_list.append(tf.stop_gradient(mruvInPath))
            input_list.append(tf.stop_gradient(mruv_confidence))
            
        # Get predicted step using decoder
        stepPath, thisHiddenState, atten_img, atten_hist = PathDecModel(input_list, training = training_state)

        # Clip gradients during backward pass
        stepPath = clip_gradients_01(stepPath)

        # Get predicted path coordinates
        if increment_net:
            thisPath = nextPath+stepPath
        else:
            thisPath = stepPath

        # Add out step
        out_list.append(thisPath)

        # Save attention outputs
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
    elif mruv_guiding and training_state:
        return out_list, mruv_v, mruv_a, mruv_confidence    
    else:
        return out_list


@tf.function
def modelV1_forward_pass(img_t, hist_t, histAvail, stepsInfer, ImageEncModel, HistEncModel, PathDecModel,
                       thisHiddenState = None,
                       use_teacher_force = False, 
                       teacher_force_weight = tf.constant(1.0), 
                       target_path=None,
                       stop_gradient_on_prediction = False,
                       gradient_clip_thorugh_time_value = 10.0,
                       return_attention = False,
                       training_state=False):
    """
    Forward pass of the path decoding model V1.
    
    This function applies all models to get features and the number of requested
    predictions.
    
    Args:
        img_t (tf.tensor / np.array): Tensor with input map batch, multi channel image shape: [batch, x, y, ch].
        hist_t (tf.tensor / np.array): Path history tensor with shape: [batch, hist_steps, coords].
        HistAvail  (tf.tensor / np.array): Tensor with history path availability (boolean), shape: [batch, steps, coords].
        stepsInfer (np.int): Number of future steps to predict.
        ImageEncModel (keras.model): Keras model for image encoding.
        HistEncModel  (keras.model): Keras model for history path encoding.
        PathDecModel (keras.model): Keras model for path decoding/generation.
        initial_hidden_state (np.ndarray): Initial state of the path decoder RNN (used only in V1).
        use_teacher_force (np.bool): Whether or not use teacher force during training.
        teacher_force_weight (np.float): How much does the teacher force modifies the RNN output.
        target_path (tf.tensor / np.array): Tensor with objective path, shape: [batch, steps, coords].
        stop_gradient_on_prediction (np.bool): Whether or not stop the gradient from flowing back through the RNN steps.
        gradient_clip_thorugh_time_value (np.float): Value of the gradient clipping during the backward pass, not used because it is not supported in a tf.function, the clip value must be fixed.
        return_attention (np.bool): Return attention weights.
        training_state (np.bool): True -> Training, False -> Inference.
          
    Returns:
        out_list (tf.tensor): Predicted path points.
    """
                       
    # Reset the hidden states for this batch 
    PathDecModel.reset_states()
    HistEncModel.reset_states()
    
    # Number of history states
    stepsHist = hist_t.shape[1]
    #stepsHist = tf.reduce_sum(histAvail, axis=-1)
    
    if not tf.is_tensor(thisHiddenState):
        thisHiddenState = tf.zeros(PathDecModel.inputs[-1].shape)
    
    # Process input image
    img_feats = ImageEncModel(img_t, training = training_state)  

    # Process history
    for idx_step_hist in range(stepsHist-1):
        hist_outs, hist_feats = HistEncModel(tf.expand_dims(hist_t[:,stepsHist-1-idx_step_hist,:], 1), training = training_state)
        
        hist_outs = tf.squeeze(hist_outs)
        hist_feats = tf.squeeze(hist_feats)

        # Get predicted path using decoder 
        thisPath, thisHiddenState, _ = PathDecModel([img_feats, hist_outs, hist_feats, thisHiddenState], training = training_state)

        thisPath = clip_gradients_01(thisPath)
        thisHiddenState = clip_gradients_01(thisHiddenState)
        hist_outs = clip_gradients_01(hist_outs)
        hist_feats = clip_gradients_01(hist_feats)
        
    
    
    # Output List
    out_list = list()

    # Create requested path points
    nextPath = hist_t[:,0,:] # Process current step first
    for idx_step in range(stepsInfer):
        
        # Update history encoding
        nextPath = tf.expand_dims(nextPath, 1)
        if stop_gradient_on_prediction:
            nextPath = tf.stop_gradient(nextPath)
        hist_outs, hist_feats = HistEncModel(nextPath, training = training_state)

        hist_outs = tf.squeeze(hist_outs)
        hist_feats = tf.squeeze(hist_feats)

        # Get predicted step using decoder
        thisPath, thisHiddenState, _ = PathDecModel([img_feats, hist_outs, hist_feats, thisHiddenState], training = training_state)

        img_feats = clip_gradients_01(img_feats)
        hist_outs = clip_gradients_01(hist_outs)
        hist_feats = clip_gradients_01(hist_feats)
        thisPath = clip_gradients_01(thisPath)
        thisHiddenState = clip_gradients_01(thisHiddenState)
        
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



@tf.function
def modelBaseline_forward_pass(img_t, ImageEncModel, PathDecModel, training_state=False, **kwargs):
    """
    Forward pass of the lyft basline. 

    Args:
        img_t (tf.tensor / np.array): Tensor with input map batch, multi channel image shape: [batch, x, y, ch].
        ImageEncModel (keras.model): Keras model for image encoding.
        PathDecModel (keras.model): Keras model for path decoding/generation.
        training_state (np.bool): True -> Training, False -> Inference.
    Returns:
        out_list (tf.tensor): Predicted path points.
    """

    # Process input image
    img_feats = ImageEncModel(img_t, training = training_state)  
    # Get Path
    out_list = PathDecModel(img_feats, training = training_state)

    return out_list


###############################################################################
# ------------------------ PATH PREDICTION DECODING NETS -------------------- #
###############################################################################


def pathDecoderModel_Baseline(cfg, ImageEncModel):
    """
    Creates a Keras model for Agent path prediction based on a convolutional network.
    Following lyft baseline.
    
        
    Args:
        cfg (dict): Configuration dictionary.
        ImageEncModel (keras.model): Keras image encoding model.
        
    Returns:
        PathDecModel (keras.model): Keras path decoding model.
    """

    # Get parameters from config
    num_path_decode_fc_layers  = cfg["model_params"]["num_path_decode_fc_layers"]
    path_decode_fc_units       = cfg["model_params"]["path_decode_fc_units"]
    path_decode_fc_activation  = cfg["model_params"]["path_decode_fc_activation"]
    use_angle                  = cfg["model_params"]["use_angle"]
    future_num_frames          = cfg["model_params"]["future_num_frames"]

    # Set number of coordinates to infer
    if use_angle:
        num_outs = 3
    else:
        num_outs = 2
    # get number of pixels and features per pixel of the image feature map
    imageProcPixelNum          = ImageEncModel.output_shape[1]
    imageProcFeatNum           = ImageEncModel.output_shape[2]

    
    # Define inputs
    Input_Image_Features = keras.Input(shape=(imageProcPixelNum, imageProcFeatNum), 
                                        name="Input_Image_Features")

    # Global average on image features
    featOutT = tf.keras.layers.GlobalAveragePooling1D()(Input_Image_Features)

    # Get predicted position
    for idx_decode_layer in range(num_path_decode_fc_layers):
        featOutT = keras.layers.Dense(path_decode_fc_units[idx_decode_layer],
                                    activation = path_decode_fc_activation,
                                    name='pathOutHid_%d'%idx_decode_layer)(featOutT)
    # Last output
    pathOut = keras.layers.Dense(future_num_frames*num_outs,
                                activation = None,
                                name='pathOutLinear')(featOutT)

    # shape [batch, steps, coords]
    pathOut = tf.reshape(pathOut, shape=[-1, future_num_frames, num_outs])

    # Build model and return
    return keras.Model(Input_Image_Features, pathOut, name='Path_Decoder_Model')


def pathDecoderModel_V2(cfg, ImageEncModel, HistEncModel):
    """
    Creates a Keras model for Agent path prediction based on a given recurrent unit.
    
    This model takes as input the code tensors from the encoding models and outputs 
    a predicted path. This models implements attention on the image code tensor and
    on the states of a previous recurrent unit used to encode the history path.
    Optionally it accepts an additional coordinate from an auxiliary method such as
    the uniform linear movement of the agent.
        
    Args:
        cfg (dict): Configuration.
        ImageEncModel (keras.model): Keras image encoding model.
        HistEncModel (keras.model): Keras path history encoding model.
    Returns:
        PathDecModel (keras.model): Keras path decoding model.
    """

    with tf.name_scope("PathDecodingModel"):

        # Get config
        gen_batch_size             = cfg["train_data_loader"]["batch_size"]
        histEnc_recurrent_unit_num = cfg["model_params"]["history_encoder_recurrent_units_number"]
        pathDec_recurrent_unit     = cfg["model_params"]["path_generation_decoder_recurrent_unit"]
        pathDec_recurrent_unit_num = cfg["model_params"]["path_generation_decoder_recurrent_units_number"]
        pathDec_attention_pr_units = cfg["model_params"]["pathDec_attention_pr_units"]
        num_path_decode_fc_layers  = cfg["model_params"]["num_path_decode_fc_layers"]
        path_decode_fc_units       = cfg["model_params"]["path_decode_fc_units"]
        path_decode_fc_activation  = cfg["model_params"]["path_decode_fc_activation"]
        path_noise_level           = cfg["model_params"]["path_noise_level"]
        pathDec_use_attention_hist = cfg["model_params"]["pathDec_use_attention_hist"]
        pathDec_use_attention_img  = cfg["model_params"]["pathDec_use_attention_img"]
        use_angle                  = cfg["model_params"]["use_angle"]
        mruv_guiding               = cfg["model_params"]["mruv_guiding"]

        # Set number of coords to infer
        if use_angle:
            num_outs = 3
        else:
            num_outs = 2
        # Get image encoding size
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
        Input_position = keras.Input(batch_shape=(gen_batch_size, num_outs), name="Input_position")
        if mruv_guiding:
            Input_mruv = keras.Input(batch_shape=(gen_batch_size, num_outs), name="Input_mruv")
            Input_mruv_conf = keras.Input(batch_shape=(gen_batch_size, num_outs), name="Input_mruv_conf")

        # Add noise to input
        in_position = tf.keras.layers.GaussianNoise(path_noise_level)(Input_position)

        # Set image feature input mechanism
        if pathDec_use_attention_img:
            with tf.name_scope("ImageAttention"):
                # Set attention mechanism for image
                imageAttnFeatT, attenScores_image = BahdanauAttention(pathDec_attention_pr_units, 
                                                                    name='BahdanauImageAttention')([Input_Last_Hidden_State,
                                                                                                    Input_Image_Features])

                # Project the physical attention to a lower space (matching path history encoding)
                imageAttnFeatT = keras.layers.Dense(histEnc_recurrent_unit_num,
                                                    kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                                    activation=keras.layers.LeakyReLU(alpha=0.1),
                                                    name='imgAttnProjection')(imageAttnFeatT)
        else:
            with tf.name_scope("ImagePooling"):  
                imageAttnFeatT = tf.keras.layers.GlobalAveragePooling1D()(Input_Image_Features)
                attenScores_image = tf.ones_like(imageAttnFeatT)

        # Set history path feature input mechanism
        if pathDec_use_attention_img:
            with tf.name_scope("HistoryAttention"):            
                # Set attention mechanism for history
                histAttnFeatT, attenScores_history = BahdanauAttention(pathDec_attention_pr_units,
                                                                    name='BahdanauHistoryAttention')([Input_Last_Hidden_State, 
                                                                                                        Input_History_Out])
        else:
            with tf.name_scope("HistoryPooling"):  
                histAttnFeatT = tf.keras.layers.GlobalAveragePooling1D()(Input_History_Out)
                attenScores_history = tf.ones_like(histAttnFeatT)


            
        # Generation of next path
        with tf.name_scope("PathGeneration"):           

            # Concatenate feature tensors with current position
            feature_list = [imageAttnFeatT, histAttnFeatT, in_position]
            if mruv_guiding:
                feature_list.append(Input_mruv)
                feature_list.append(Input_mruv_conf)
            featT = keras.layers.Concatenate(axis=-1, name='pathFeatConcat')(feature_list)


            # Get recurrent unit to use
            try:
                pathDec_base_recurrent_unit = getattr(keras.layers, pathDec_recurrent_unit)
            except:
                raise Exception('Path decoder base recurrent unit not found. Requested unit: %s'%pathDec_recurrent_unit)

            # Apply recurrent output
            featT = K.expand_dims(featT, 1)
            rec_outs = pathDec_base_recurrent_unit(pathDec_recurrent_unit_num,
                                                   kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                                   recurrent_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                                   stateful=True,
                                                   return_state=True,
                                                   name='pathRecUnit')(featT)
            # Get outputs and state
            featOutT = rec_outs[0]
            featHidT = rec_outs[-1]

            # Get predicted position
            for idx_decode_layer in range(num_path_decode_fc_layers):
                featOutT = keras.layers.Dense(path_decode_fc_units[idx_decode_layer],
                                              kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                              activation = path_decode_fc_activation,
                                              name='pathOutHid_%d'%idx_decode_layer)(featOutT)
            # Last output
            pathOut = keras.layers.Dense(num_outs,
                                        activation = None,
                                        name='pathOutLinear')(featOutT)

        # Define inputs with or without mruv guiding
        input_list = [Input_position, 
                      Input_Image_Features,
                      Input_History_Out, 
                      Input_History_Hidden, 
                      Input_Last_Hidden_State]
        if mruv_guiding:
            input_list.append(Input_mruv)
            input_list.append(Input_mruv_conf)

        # Create model
        return keras.Model(input_list,
                           [pathOut, featHidT, attenScores_image, attenScores_history],
                           name='Path_Decoder_Model')


def pathDecoderModel_V1(cfg, ImageEncModel, HistEncModel):
    """
    Creates a Keras model for Agent path prediction based on a given recurrent unit.
    
    This model takes as input the code tensors from the encoding models and outputs 
    a predicted path. This models implements attention on the image code tensor.
    
    Args:
        cfg (dict): Configuration.
        ImageEncModel (keras.model): Keras image encoding model.
        HistEncModel (keras.model): Keras path history encoding model.
    Returns:
        PathDecModel (keras.model): Keras path decoding model.
    """

    # Get config
    gen_batch_size             = cfg["train_data_loader"]["batch_size"]
    histEnc_recurrent_unit_num = cfg["model_params"]["history_encoder_recurrent_units_number"]
    pathDec_recurrent_unit     = cfg["model_params"]["path_generation_decoder_recurrent_unit"]
    pathDec_recurrent_unit_num = cfg["model_params"]["path_generation_decoder_recurrent_units_number"]
    pathDec_attention_pr_units = cfg["model_params"]["pathDec_attention_pr_units"]
    num_path_decode_fc_layers  = cfg["model_params"]["num_path_decode_fc_layers"]
    path_decode_fc_units       = cfg["model_params"]["path_decode_fc_units"]
    path_decode_fc_activation  = cfg["model_params"]["path_decode_fc_activation"]
    use_angle                  = cfg["model_params"]["use_angle"]

    if use_angle:
        num_outs = 3
    else:
        num_outs = 2

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


    # Set attention mechanism for image
    with tf.name_scope("ImageAttention"):
        Image_query = Input_Last_Hidden_State
        imageAttenFeatT, attenScores = BahdanauAttention(pathDec_attention_pr_units,
                                                         name='BahdanauImageAttention')([Image_query, Input_Image_Features])

    with tf.name_scope("PathGeneration"): 
        # Concatenate feature tensors
        featT = keras.layers.Concatenate(axis=-1)([imageAttenFeatT, Input_History_Out])


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
                                                          stateful=True,
                                                          return_state=True,
                                                          name='pathRecUnit')(featT)

        # Get predicted position
        for idx_decode_layer in range(num_path_decode_fc_layers):
            featOutT = keras.layers.Dense(path_decode_fc_units[idx_decode_layer],
                                          kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                          activation = path_decode_fc_activation,
                                          name='pathOutHid_%d'%idx_decode_layer)(featOutT)
        # Last output
        pathOut = keras.layers.Dense(num_outs,
                                    activation = None,
                                    name='pathOutLinear')(featOutT)


    # Create model
    return keras.Model([Input_Image_Features,
                        Input_History_Out, 
                        Input_History_Features, 
                        Input_Last_Hidden_State],
                       [pathOut, featHidT, attenScores],
                       name='Path_Decoder_Model')



###############################################################################
# ------------------------ LINEAR MOVEMENT NET ------------------------------ #
###############################################################################

def mruvModel(cfg, ImageEncModel):
    """
    Creates a Keras model to infer the speed and acceleration of an Agent.
    It also estimates a confidence value of the linear movement prediction.
        
    Args:
        cfg (dict): Configuration.
        ImageEncModel (keras.model): Keras image encoding model.
    Returns:
        PathDecModel (keras.model): Keras path decoding model.
    """

    # Get config
    histEnc_recurrent_unit_num  = cfg["model_params"]["history_encoder_recurrent_units_number"]
    num_mruv_layers             = cfg["model_params"]["num_mruv_layers"]
    mruv_units                  = cfg["model_params"]["mruv_units"]
    use_angle                   = cfg["model_params"]["use_angle"]

    # Define an activation function
    mruv_activation             = keras.layers.LeakyReLU(alpha=0.1)
    # Get image encoding feature map size
    imageProcPixelNum          = ImageEncModel.output_shape[1]
    imageProcFeatNum           = ImageEncModel.output_shape[2]

    # Set coordinates dimension
    if use_angle:
        num_outs = 3
    else:
        num_outs = 2


    with tf.name_scope("mruvModel"):
        # Inputs
        Input_mruv_hist_feats = keras.Input(shape=(histEnc_recurrent_unit_num), 
                                        name="Input_history_features")
        Input_mruv_image_feats = keras.Input(shape=(imageProcPixelNum, imageProcFeatNum), 
                                        name="Input_image_features")

        # Global average on image features
        imageFeatT = tf.keras.layers.GlobalAveragePooling1D()(Input_mruv_image_feats)
        imageFeatT = tf.ones_like(imageFeatT)

        # Concatenate image and history features
        mruv_feats = keras.layers.Concatenate(axis=-1)([Input_mruv_hist_feats, imageFeatT])

        # Apply hidden layers
        for idx_layer in range(num_mruv_layers):
            mruv_feats = keras.layers.Dense(mruv_units[idx_layer],
                                        name='mruvHid_%d'%idx_layer,
                                        activation=mruv_activation)(mruv_feats)
        # Output layer
        mruvOut = keras.layers.Dense(num_outs*3,
                                    activation = None,
                                    name='mruvOutLinear')(mruv_feats)

        # Divide into speed, acceleration and confidence
        vOut, aOut, confidenceOut = tf.split(mruvOut, 3, -1)

        # Create model
        return keras.Model([Input_mruv_hist_feats, Input_mruv_image_feats], [vOut, aOut, confidenceOut], name='mruv_Model')




###############################################################################
# ------------------------ PATH HISTORY ENCODING NET ------------------------ #
###############################################################################

def pathEncodingModel(cfg, return_sequences=False):
    """
    Creates a Keras model for Agent history encoding based on a given recurrent unit.
    
    Args:
        cfg (dict): Configuration.
        return_sequences (np.bool): If true the outputs and states for each input step are returned       
    Returns:
        HistEncModel (keras.model): Keras path history encoding model.
    """
    with tf.name_scope("HistoryEncodingModel"):
        # Get config
        gen_batch_size             = cfg["train_data_loader"]["batch_size"]
        histEnc_recurrent_unit     = cfg["model_params"]["history_encoder_recurrent_unit"]
        histEnc_recurrent_unit_num = cfg["model_params"]["history_encoder_recurrent_units_number"]
        num_hist_encode_layers     = cfg["model_params"]["num_hist_encode_layers"]
        hist_encode_units          = cfg["model_params"]["hist_encode_units"]
        path_noise_level           = cfg["model_params"]["path_noise_level"]
        CarNet                     = cfg["model_params"]["CarNet"]
        use_angle                  = cfg["model_params"]["use_angle"]

        # Define activation function
        hist_encode_activation     = keras.layers.LeakyReLU(alpha=0.1)

        # set number of input coordinates
        if use_angle:
            num_ins = 3
        else:
            num_ins = 2

        # Inputs
        Input_hist_frames = keras.Input(batch_shape=(gen_batch_size, 1, num_ins), 
                                        name="Input_history_frames")

        # Add noise
        pathFeatT = tf.keras.layers.GaussianNoise(path_noise_level)(Input_hist_frames)

        # History encoding dense layers
        for idx_encode_layer in range(num_hist_encode_layers):
            pathFeatT = keras.layers.Dense(hist_encode_units[idx_encode_layer],
                                        kernel_regularizer=keras.regularizers.l1_l2(0.01,0.01),
                                        activation=hist_encode_activation)(pathFeatT)

        # If this is not the CarNet, use a recurrent unit here
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
                                                return_state=True,
                                                return_sequences=return_sequences,
                                                stateful=True,
                                                name='histRecUnit')(pathFeatT)
            # Get output and states
            encoder_out = outs[0]
            encoder_states = outs[-1]
        else:
            pathFeatT = keras.layers.Dense(histEnc_recurrent_unit_num,
                                        activation=hist_encode_activation)(pathFeatT)

            # Return the encoding of the input coordinates
            encoder_out = pathFeatT
            encoder_states = pathFeatT

        

        return keras.Model(Input_hist_frames, [encoder_out, encoder_states], name='History_Encoding_Model')


###############################################################################
# ------------------------ IMAGE ENCODING ----------------------------------- #
###############################################################################

def imageEncodingModel(base_img_model, cfg):
    """
    Creates a Keras model for image encoding based on a given convolutional network.
    
    Args:
        base_img_model (keras.model): Keras base image encoding model.
        cfg (dict): Configuration.
    Returns:
        ImageEncModel (keras.model): Keras image encoding model.
    """
    
    # Get config
    model_map_input_shape      = (cfg["raster_params"]["raster_size"][0],
                                  cfg["raster_params"]["raster_size"][1])
    retrain_inputs_image_model = cfg["training_params"]["retrain_inputs_image_model"]
    retrain_all_image_model    = cfg["training_params"]["retrain_all_image_model"]
    history_num_frames         = cfg["model_params"]["history_num_frames"]
    use_fading                 = cfg["model_params"]["use_fading"]

    retrain_inputs_image_model = retrain_all_image_model or retrain_inputs_image_model

    # Get number of input image channels 
    if use_fading:
        map_input_channels         = 3+2 # RGB + Ego Fade + Agent Fade
    else:
        map_input_channels         = 3+(history_num_frames*2)+2 # RGB + (1 per each history frame + 1 current) x 2
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
          
    with tf.name_scope("ComposedInputLayer"):
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
        # Apply
        mapT = newConv(mapT)

        # Set RGB filter channels to the pretrained values
        # For the new channels we will copy the red channel 
        # (most important for human vision, maybe also in this net?, nevertheless we will retrain them if we want...)
        this_weights = layer_start.get_weights()
        new_weights = np.zeros((layer_start.kernel_size[0], layer_start.kernel_size[1], map_input_channels, layer_start.filters))
        new_weights[:,:,:3,:] = this_weights[0]
        for idx_newChn in range(map_input_channels-this_weights[0].shape[2]):
            new_weights[:,:,this_weights[0].shape[2]+idx_newChn,:] = this_weights[0][:,:,0,:]
        # bias
        if len(this_weights) > 1:
            new_bias = this_weights[1]
            new_weights = [new_weights, new_bias]
        else:
            new_weights = [new_weights]
        newConv.set_weights(new_weights)



    with tf.name_scope("BaseNetLayers"):
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

    with tf.name_scope("OutputLayers"):
        # Reshape feature map
        mapFeatT = K.reshape(mapT, shape=[-1,mapT.shape[1]*mapT.shape[2],mapT.shape[3]])

    return keras.Model(Input_map, mapFeatT, name='Image_Encodding_Model')

    


###############################################################################
# ------------------------ CUSTOM LAYERS AND FUNCTIONS ---------------------- #
###############################################################################

# Custom gradient clip for RNN
@tf.custom_gradient
def clip_gradients_01(y):
    """
    Identity function with clipped gradient during backward pass.
    Used in the RNN steps to clip the gradient during its calculation.

    Args:
        y (tf.tensor): input
    Args:
        y (tf.tensor): the same as input
    """
    def backward(dy):
        return tf.clip_by_norm(dy, 0.01)
    return y, backward


# Attention layer from https://www.tensorflow.org/tutorials/text/nmt_with_attention
# (with some modifications for save/load operations)
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(self.units, use_bias=False, name='W1')
        self.W2 = tf.keras.layers.Dense(self.units, use_bias=False, name='W2')
        self.V = tf.keras.layers.Dense(1, use_bias=False, name='V')

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
    """
    Resets the state of the "statefull" recurrent layers within a model.
    Optionally sets the states to a given value.

    ------------> Not working in graph mode... for some reason...

    Args:
        model (keras.model): Model to be resetted.
        layer_states_dict (dict): Dictionary containing the name and value of each recurrent layer to be initialized.
    """
    for layer in model.layers:
        if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):

            if layer.name in layer_states_dict:
                layer.reset_states(states = layer_states_dict[layer.name])
            else:
                layer.reset_states()

    return