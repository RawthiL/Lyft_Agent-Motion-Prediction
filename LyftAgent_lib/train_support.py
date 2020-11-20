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
from l5kit.configs import load_config_data



###############################################################################
# ------------------------ LOSS FUNCTIONS ----------------------------------- #
###############################################################################


def tf_neg_multi_log_likelihood(
    ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    For more details about used loss function and reformulation, please see
    https://github.com/lyft/l5kit/blob/master/competition.md.
    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: negative log-likelihood for this example, a single float number
    """

    ground_truth = tf.expand_dims(ground_truth,  axis=0)  # add modes
    avails = avails[np.newaxis, :, np.newaxis]  # add modes and cords

    error = tf.reduce_sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = tf.math.log(confidences) - 0.5 * tf.reduce_sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = tf.reduce_max(error)  # error are negative at this point, so max() gives the minimum one
    error = -tf.math.log(tf.reduce_sum(tf.exp(error - max_value), axis=-1)) - max_value  # reduce modes
    return error




@tf.function
def L_loss_single2mult(objPath, predPAth, availPoints):
    """
    Uses "tf_neg_multi_log_likelihood" to compute the likelihood of a batch
    of samples.

    Args:
        objPath (np.ndarray): Objective/Real path
        predPAth (np.ndarray): Estimated path.
        availPoints (np.ndarray): Availability of each datapoint (boolean)
    Returns:
        out_L (np.ndarray): Likelihood for each path
    """

    out_L = list()
    # Process each sample in batch
    for idx_batch in range(objPath.shape[0]):
        valLikelihood = tf_neg_multi_log_likelihood(objPath[idx_batch,:,:2],
                                                                    tf.expand_dims(predPAth[idx_batch,:,:2], axis=0),
                                                                    1.0,
                                                                    availPoints[idx_batch,:])

        out_L.append(valLikelihood)

    # Convert results list to tensor
    out_L = tf.convert_to_tensor(out_L)

    return out_L



@tf.function
def L2_loss(objPath, predPAth, availPoints):
    """
    Calculate the mean L2 loss of the prediction.

    Args:
        objPath (np.ndarray): Real path.
        predPAth (np.ndarray): Predicted path.
        availPoints (np.ndarray): Number valid real points.

    Returns:
        out_loss (np.ndarray): Mean L2 loss.
    """

    # Mask with availability
    predPAth = tf.multiply(predPAth, tf.tile(tf.expand_dims(availPoints,axis=-1), [1,1,predPAth.shape[-1]]))
    objPath = tf.multiply(objPath, tf.tile(tf.expand_dims(availPoints,axis=-1), [1,1,predPAth.shape[-1]]))
    # L2
    out_loss = tf.reduce_sum((predPAth - objPath)**2, axis = -1)
    out_loss = tf.reduce_sum(out_loss, axis = -1)

    # Get mean error for each sample in batch
    stepsInfer_byBatch = tf.reduce_sum(availPoints, axis = -1)

    return tf.divide(out_loss, stepsInfer_byBatch)


def tf_diff_along_step(a):
    """
    Calculates the numerical derivate of the input tensor.
    Only works on path tensors of shape [batch, steps, coords]

    Args:
        a (tf.tensor): input vehicle path.
    Returns:
        diff (tf.tensor): Derivate of "s", shape [batch, steps-1, coords]
    """
    # Shift and substract the tensor agaisnt itself
    diff = (tf.roll(a, shift=1, axis=1) - a)
    # Return all but first sample
    return diff[:,1:,:]

def get_velocity_and_acceleration(batch_pos):
    """
    Calculates the velocity and acceleration of a batch of vehicles.

    Args:
        batch_pos (tf.tensor): Batch of positions, shqpe [batch, steps, coords]
    Returns:
        v (tf.tensor): Velocity of each vehicle.
        a (tf.tensor): Acceleration of each vehicle.
        confidence (tf.tensor): Confidence of calculation based on the deviation of the mean.
    """

    # Get first derivate
    v_inst = tf_diff_along_step(batch_pos)
    # Get the velocity as the mean of the first derivate
    v = tf.reduce_mean(v_inst, axis = 1)
    # Calculate the deviation of the mean for the velocity
    std_v = tf.math.reduce_std(v_inst, axis = 1)
    v_mean_norm_conf = tf.divide((std_v/tf.sqrt(tf.cast(batch_pos.shape[1], dtype=tf.float32))),v)
    v_mean_norm_conf = tf.abs(v_mean_norm_conf)

    # Get second derivate
    a_inst = tf_diff_along_step(v_inst)
    # Get the acceleration as the mean of the second derivate
    a = tf.reduce_mean(a_inst, axis = 1)
    # Calculate the deviation of the mean for the acceleration
    std_a = tf.math.reduce_std(a_inst, axis = 1)
    a_mean_norm_conf = tf.divide((std_a/tf.sqrt(tf.cast(batch_pos.shape[1], dtype=tf.float32))),a)
    a_mean_norm_conf = tf.abs(a_mean_norm_conf)

    # Build a confidence tensor
    confidence = 1.0-tf.clip_by_value(v_mean_norm_conf*a_mean_norm_conf, 0.0, 1.0)

    return v, a, confidence

###############################################################################
# ------------------------ CUSTOM TRAIN FUNCTIONS --------------------------- #
###############################################################################

@tf.function
def generator_train_step_Base(img_t, target_path, TargetAvail, ImageEncModel, PathDecModel,
                             optimizer_gen,
                             gen_loss,
                             forwardpass_use,
                             loss_use = [L2_loss],
                             loss_couplings = [1],
                             gradient_clip_value = 10.0):
    """
    Train step function of the lyft base model.

    Args:
        img_t (tf.tensor / np.array): Tensor with input map batch, multi channel image shape: [batch, x, y, ch].
        target_path (tf.tensor / np.array): Tensor with objective path, shape: [batch, steps, coords].
        TargetAvail (tf.tensor / np.array): Tensor with objective path availability (boolean), shape: [batch, steps, coords].
        ImageEncModel (keras.model): Keras model for image encoding.
        PathDecModel (keras.model): Keras model for path decoding/generation.
        optimizer_gen (keras.optimizer): Keras optimizer... for training...
        gen_loss (keras.metric): Keras metric placeholder, not a real use for this...
        forwardpass_use (tf.function): Function which receives the inputs and applies the models.
        loss_use (list): List of losses to apply, defaults to L2.
        loss_couplings (list): Couplings of the losses, this list must have the same length as "loss_use" and be filled with floats.
        gradient_clip_value (np.float): Maximum allowed gradient norm.
    Returns:
        gen_step_loss_list (list): List of loss tensors of this train step.
        out_grad (list): List of gradients of this train step.
    """

    stepsInfer = PathDecModel.output.shape[1]

    # Start gradient tape
    with tf.GradientTape(persistent=True) as gen_tape:

        # Get predicted paths
        with tf.name_scope("Forward_pass"):
            thisPath = forwardpass_use(img_t, ImageEncModel, PathDecModel, training_state = True)

        with tf.name_scope("Loss"):
            # Compute predicted path loss
            gen_step_loss_list = list()
            gen_step_loss = 0
            for loss_this, k_coup in zip(loss_use, loss_couplings):
                gen_step_loss_list.append(loss_this(target_path[:,:stepsInfer,:], thisPath, TargetAvail[:,:stepsInfer]))
                gen_step_loss += k_coup*gen_step_loss_list[-1]

    with tf.name_scope("Gradient_calc"):
        # Gradient wrt ImageEncModel
        image_model_trainable =  len(ImageEncModel.trainable_variables)>0
        if image_model_trainable:
            grad_gen_img = gen_tape.gradient(target  = gen_step_loss,
                                            sources = ImageEncModel.trainable_variables)
        # Gradient wrt PathDecModel
        grad_gen_dec  = gen_tape.gradient(target  = gen_step_loss,
                                        sources = PathDecModel.trainable_variables)
        # Delete persistent gradient
        del gen_tape

        # We now clip the recursive models
        if image_model_trainable:
            grad_gen_img, _ = tf.clip_by_global_norm(grad_gen_img, gradient_clip_value)
        grad_gen_dec, _ = tf.clip_by_global_norm(grad_gen_dec, gradient_clip_value)

        # Get trainable weights
        train_vars = list()
        if image_model_trainable:
            train_vars += ImageEncModel.trainable_variables
        train_vars += PathDecModel.trainable_variables
        grad_gen_clip = list()
        if image_model_trainable:
            grad_gen_clip += grad_gen_img
        grad_gen_clip += grad_gen_dec

        # Apply update
        optimizer_gen.apply_gradients(zip(grad_gen_clip, train_vars))

    # Log gradient norms for each model
    out_grad = list()
    if image_model_trainable:
        out_grad.append(grad_gen_img)
    out_grad.append(grad_gen_dec)

    # Save loss
    gen_loss(tf.reduce_mean(gen_step_loss))

    return gen_step_loss_list, out_grad


@tf.function
def generator_train_step(img_t, hist_t, target_path, HistAvail, TargetAvail,
                         ImageEncModel, HistEncModel, PathDecModel,
                         optimizer_gen,
                         gen_loss,
                         initial_hidden_state,
                         forwardpass_use,
                         loss_use = [L2_loss],
                         loss_couplings = [1],
                         stepsInfer = 50,
                         use_teacher_force=True,
                         teacher_force_weight = tf.constant(1.0, dtype=tf.float32),
                         gradient_clip_value = 10.0,
                         gradient_clip_thorugh_time_value = 1.0,
                         stop_gradient_on_prediction = False,
                         increment_net = False,
                         mruv_guiding = False,
                         mruv_model = None,
                         mruv_model_trainable = False):
    """
    Train step function of the recurrent models (V1 and V2).

    Args:
        img_t (tf.tensor / np.array): Tensor with input map batch, multi channel image shape: [batch, x, y, ch].
        hist_t (tf.tensor / np.array): Path history tensor with shape: [batch, hist_steps, coords].
        target_path (tf.tensor / np.array): Tensor with objective path, shape: [batch, steps, coords].
        HistAvail  (tf.tensor / np.array): Tensor with history path availability (boolean), shape: [batch, steps, coords].
        TargetAvail (tf.tensor / np.array): Tensor with objective path availability (boolean), shape: [batch, steps, coords].
        ImageEncModel (keras.model): Keras model for image encoding.
        HistEncModel  (keras.model): Keras model for history path encoding.
        PathDecModel (keras.model): Keras model for path decoding/generation.
        optimizer_gen (keras.optimizer): Keras optimizer... for training...
        gen_loss (keras.metric): Keras metric placeholder, not a real use for this...
        initial_hidden_state (np.ndarray): Initial state of the path decoder RNN (used only in V1).
        forwardpass_use (tf.function): Function which receives the inputs and applies the models.
        loss_use (list): List of losses to apply, defaults to L2.
        loss_couplings (list): Couplings of the losses, this list must have the same length as "loss_use" and be filled with floats.
        stepsInfer (np.int): Number of future steps to predict.
        use_teacher_force (np.bool): Whether or not use teacher force during training.
        teacher_force_weight (np.float): How much does the teacher force modifies the RNN output.
        gradient_clip_value (np.float): Maximum allowed gradient norm.
        gradient_clip_thorugh_time_value (np.float): Value of the gradient clipping during the backward pass, not used because it is not supported in a tf.function, the clip value must be fixed.
        stop_gradient_on_prediction (np.bool): Whether or not stop the gradient from flowing back through the RNN steps.
        increment_net (np.bool): If this is an increment net the RNN only predicts the step length, not the absolute positions.
        mruv_guiding (np.bool): If true a model is used to calculate the speed and acceleration of the vehicle and infer the future steps using accelerated linear movement. This is fed to the network as an additional input.
        mruv_model (keras.model / function): A model or a function which generates the speed and acceleration for the batch.
        mruv_model_trainable (np.boolean): If the mruv_model is trainable we must get gradients out of it.

    Returns:
        gen_step_loss_list (list): List of loss tensors of this train step.
        out_grad (list): List of gradients of this train step.
    """


    # Get number of steps to generate
    # if not tf.is_tensor(stepsInfer):
    #     stepsInfer = target_path.shape[1]


    # Start gradient tape
    with tf.GradientTape(persistent=True) as gen_tape:

        # Get predicted paths
        with tf.name_scope("Forward_pass"):
            outputs_fpass = forwardpass_use(img_t, hist_t, HistAvail, stepsInfer,
                                        ImageEncModel, HistEncModel, PathDecModel,
                                        thisHiddenState = initial_hidden_state,
                                        use_teacher_force = use_teacher_force,
                                        teacher_force_weight = teacher_force_weight,
                                        target_path = target_path,
                                        stop_gradient_on_prediction = stop_gradient_on_prediction,
                                        gradient_clip_thorugh_time_value= gradient_clip_thorugh_time_value,
                                        training_state = True,
                                        increment_net = increment_net,
                                        mruv_guiding = mruv_guiding,
                                        mruv_model = mruv_model,
                                        mruv_model_trainable = mruv_model_trainable)

            # If the forward pass uses mruv guiding more outputs are expected
            if mruv_guiding:
                thisPath = outputs_fpass[0]
                thisV = outputs_fpass[1]
                thisA = outputs_fpass[2]
                thisConf = outputs_fpass[3]
            else:
                thisPath = outputs_fpass

        with tf.name_scope("Loss"):

            # Compute predicted path loss
            gen_step_loss_list = list()
            gen_step_loss = 0
            for loss_this, k_coup in zip(loss_use, loss_couplings):
                loss_act = loss_this(target_path[:,:stepsInfer,:], thisPath, TargetAvail[:,:stepsInfer])
                gen_step_loss_list.append(loss_act)
                gen_step_loss += k_coup*loss_act

            # Get loss of the mruv guiding network
            if mruv_guiding and mruv_model_trainable:
                # Calculate speed and acceleration using the derivate of the history
                targetV, targetA, _ = get_velocity_and_acceleration(hist_t)
                lossV = tf.reduce_mean((thisV - targetV)**2, axis = -1)
                gen_step_loss_list.append(lossV)
                lossA = tf.reduce_mean((thisA - targetA)**2, axis = -1)
                gen_step_loss_list.append(lossA)

                # Get expected mruv path
                mruv_path = lyl_nn.compute_mruv_path(targetV, targetA, stepsInfer)
                # Compute MSE between mruv and target
                mruv_L2 = tf.reduce_mean((mruv_path[:,1:stepsInfer+1,:] - target_path[:,:stepsInfer,:])**2, axis = 1)
                # mruv_confidence = tf.exp(mruv_L2)
                mruv_confidence = (mruv_L2/tf.norm(target_path[:,:stepsInfer,:], axis = 1))/500.0
                mruv_confidence = tf.clip_by_value(mruv_confidence, 0.0, 2.0)
                # Get confidence Loss
                lossConf = tf.reduce_sum((thisConf - mruv_confidence)**2, axis = -1)*0.001
                gen_step_loss_list.append(lossConf)

                # Final mruv model loss
                mruv_loss = lossV+lossA+lossConf


    # Compute the gradient
    with tf.name_scope("Gradient_calc"):
        # Gradient wrt ImageEncModel
        image_model_trainable =  len(ImageEncModel.trainable_variables)>0
        if image_model_trainable:
            grad_gen_img = gen_tape.gradient(target  = gen_step_loss,
                                            sources = ImageEncModel.trainable_variables)
        # Gradient wrt HistEncModel
        grad_gen_hist = gen_tape.gradient(target  = gen_step_loss,
                                        sources = HistEncModel.trainable_variables)
        # Gradient wrt PathDecModel
        grad_gen_dec  = gen_tape.gradient(target  = gen_step_loss,
                                        sources = PathDecModel.trainable_variables)

        if mruv_guiding and mruv_model_trainable:
            grad_mruv  = gen_tape.gradient(target  = mruv_loss,
                                           sources = mruv_model.trainable_variables)

        # Delete persistent gradient
        del gen_tape

        # We now clip the recursive models, the global norm!
        if image_model_trainable:
            grad_gen_img, _ = tf.clip_by_global_norm(grad_gen_img, gradient_clip_value)
        grad_gen_hist, _ = tf.clip_by_global_norm(grad_gen_hist, gradient_clip_value)
        grad_gen_dec, _ = tf.clip_by_global_norm(grad_gen_dec, gradient_clip_value)
        
        # Get full list of trainable variables and gradients
        train_vars = list()
        if image_model_trainable:
            train_vars += ImageEncModel.trainable_variables
        train_vars += HistEncModel.trainable_variables
        train_vars += PathDecModel.trainable_variables
        if mruv_guiding and mruv_model_trainable:
            train_vars += mruv_model.trainable_variables
        grad_gen_clip = list()
        if image_model_trainable:
            grad_gen_clip += grad_gen_img
        grad_gen_clip += grad_gen_hist
        grad_gen_clip += grad_gen_dec
        if mruv_guiding and mruv_model_trainable:
            grad_gen_clip += grad_mruv

        # Apply update
        optimizer_gen.apply_gradients(zip(grad_gen_clip, train_vars))


    # Log gradient norms for each model
    out_grad = list()
    if image_model_trainable:
        out_grad.append(grad_gen_img)
    out_grad.append(grad_gen_hist)
    out_grad.append(grad_gen_dec)

    if mruv_guiding and mruv_model_trainable:
        out_grad.append(grad_mruv)

    # Save loss
    gen_loss(tf.reduce_mean(gen_step_loss))

    return gen_step_loss_list, out_grad


def get_teacher_force_weight(tf_list, tf_lims, epoch, tf_weight_act, linearize=False):
    """
    Get the teacher force weight given the epoch and the weight planning.

    Args:
        tf_list (list): List of teacher force weights.
        tf_lims (list): List of teacher force limit epochs for each weight.
        epoch (np.int32): Current train epoch.
        tf_weight_act (np.float32): Current teacher force weight.
        linearize (np.bool): Linearize the teacher force weight between the given points.
    Returns:
        tf_weight_new (np.flaot32): New teacher force weight.
    """

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
    """
    Set the learning rate given the epoch and the learning rate planning.

    Args:
        lr_list (list): List of learning rates.
        lr_lims (list): List of learning rate limit epochs.
        epoch (np.int32): Current train epoch.
        optimizer (keras.optimizer): Keras optimizer.
    Returns:
        lr_new (np.float32): New learning rate.
    """

    # Get new learning rate
    lr_new = lr_list[np.squeeze(np.argwhere(np.array(lr_lims) >= epoch)[0])]
    # Update and inform if needed
    if not np.isclose(lr_new, optimizer.learning_rate.numpy(), rtol = 1e-10):
        K.set_value(optimizer.learning_rate, lr_new)
        print('Learning rate set to: %g'%optimizer.learning_rate.numpy())

    return lr_new


def get_future_steps_train(fs_list, fs_lims, epoch, fs_act):
    """
    Get the number of future steps to train on given the epoch and the planning.

    Args:
        fs_list (list): List of future steps.
        fs_lims (list): List of limit epochs.
        epoch (np.int32): Current train epoch.
        fs_act (np.float32): Current future steps weight.
    Returns:
        fs_new (np.int32) New future steps to train on.
    """

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
                   steps_validate = None, all_metrics = False, stepsInfer = -1, base_model = False,
                   mruv_guiding = False, mruv_model = None, mruv_model_trainable = False, increment_net = False):
    """
    Validate the generator model using the validation dataset.

    Args:
        tf_validation_dataset (tf.Dataset): TensorFlow Dataset object of the validation dataset.
        ImageEncModel (keras.Model): Model used to encode the input image.
        HistEncModel (keras.Model): Model used to encode the input history.
        PathDecModel (keras.Model): Model used to produce the requested predictions.
        forwardpass_use (tf.function): Function which receives the inputs and applies the models.
        steps_validate (np.int32): Number of steps to perform validation on (None means all).
        all_metrics (np.bool): Use all metrics or only L2.
        stepsInfer (np.bool): Number of future steps to validate on, -1 means all of the available in the dataset.
        base_model (np.bool): Whether or not the model is the lyft baseline.
        mruv_guiding (np.bool): Whether or not speed and acceleration estimation is being used.
        mruv_model (keras.model / function): Function or model used to estimate speed and acceleration.
        mruv_model_trainable (np.bool): If true, the mruv_model is a keras.model, otherwise it is a function.
        increment_net (np.bool): Whether or not the model is an increment RNN.
    Returns:
        mean_L2_loss (np.float) Mean L2 validation loss.
        mean_Likelihood_loss (np.float) Mean likelihood validation loss. (if all_metrics == True)
        mean_TD_loss (np.ndarray) Mean Time Displace validation loss. (if all_metrics == True)
    """

    # Setup
    idx_val = 0
    valLoss_acum = 0

    valLikelihood_acum = 0
    valTime_displace_acum = 0

    gen_batch_size = PathDecModel.inputs[0].shape[0]

    # Iterate over the validation dataset
    custom_steps = True
    val_dataset_prog_bar = tqdm(tf_validation_dataset, total=steps_validate)
    for (valSampleMapComp, valSampeHistPath, valSampeTargetPath,
             valHistAvail, valTargetAvail,
             valTimeStamp, valTrackID, valRasterFromAgent, valWorldFromAgent, valCentroid, valIDX) in val_dataset_prog_bar:

        # If the batch size of this minibatch is smaller than the expected stop (RNN models have fixed batchsize due to being statefull)
        if gen_batch_size != None:
            if valSampleMapComp.shape[0] < gen_batch_size:
                break
            
        # Get number of steps to infer if not defined
        if stepsInfer == -1:
            custom_steps = False
            stepsInfer = valSampeTargetPath.shape[-2]

        # Predict
        if base_model:
            valPredPath = forwardpass_use(valSampleMapComp,
                                                    ImageEncModel,  PathDecModel,
                                                    use_teacher_force=False)
        else:
            PathDecModel.reset_states()
            HistEncModel.reset_states()
            valPredPath = forwardpass_use(valSampleMapComp, valSampeHistPath, valHistAvail, stepsInfer,
                                                    ImageEncModel, HistEncModel, PathDecModel,
                                                    use_teacher_force=False,
                                                    increment_net = increment_net,
                                                    mruv_guiding = mruv_guiding,
                                                    mruv_model = mruv_model,
                                                    mruv_model_trainable = mruv_model_trainable)

        valPredPath = valPredPath.numpy()
        # Calculate loss
        valLoss = L2_loss(valSampeTargetPath[:,:stepsInfer,:], valPredPath, valTargetAvail[:,:stepsInfer])
        valLoss = np.mean(valLoss.numpy())

        # Update
        valLoss_acum += valLoss


        # Process lyft metrics
        if all_metrics:
            Time_displace_batch = 0
            for idx_batch in range(valPredPath.shape[0]):

                Time_displace_batch += metrics.time_displace(valSampeTargetPath[idx_batch,:stepsInfer,:2],
                                                                        np.expand_dims(valPredPath[idx_batch,:stepsInfer,:2], axis=0),
                                                                        np.ones((1)),
                                                                        valTargetAvail[idx_batch,:stepsInfer])

            valTime_displace_acum += Time_displace_batch/valPredPath.shape[0]

        valLikelihood_acum += np.mean(L_loss_single2mult(valSampeTargetPath[:,:stepsInfer,:], valPredPath, valTargetAvail[:,:stepsInfer]))


        # Update progress bar
        if all_metrics:
            msg_string = 'Validation: L2 = %.2f ; L = %.2f ; TD(T) = %0.2f '%( (valLoss_acum/(idx_val+1)),
                                                                            (valLikelihood_acum/(idx_val+1)),
                                                                            (valTime_displace_acum[-1]/(idx_val+1)))
        else:
            msg_string = 'Validation: L2 Loss: %.2f (last %.2f) '%(valLoss_acum/(idx_val+1), valLoss)

        if custom_steps:
            msg_string += ' (on %d steps)'%stepsInfer

        val_dataset_prog_bar.set_description(msg_string)

        # Update steps
        idx_val += 1
        if steps_validate != None:
            if idx_val > steps_validate:
                break

    if steps_validate != None:
        if idx_val < steps_validate:
            print('The dataset finished before the requested number of steps was reached.')

    if all_metrics:
        return valLoss_acum/idx_val, valLikelihood_acum/idx_val, valTime_displace_acum/idx_val
    else:
        return valLoss_acum/idx_val





###############################################################################
# ------------------------ TF DATASET READER -------------------------------- #
###############################################################################

def meta_dict_pass(dataset, **kwargs):
    """
    Dataset wrapper for TensorFlow Data compatibility

    Yields the dataset without any order

    Args:
        dataset (AgentDataset): Agent dataset from Lyft library
    """

    for sample_index, frame in enumerate(dataset):
        frame['sample_idx'] = sample_index
        yield frame

def meta_dict_gen(dataset,
                  randomize_frame=False,
                  randomize_scene=False,
                  num_scenes=16000,
                  frames_per_scene = 1,
                  yield_only_large_distances = False,
                  min_max_dist = 25.0):
    """
    Dataset generator wrapper for TensorFlow Data compatibility

    Yields a fixed number of scenes and frames

    Args:
        dataset (AgentDataset): Agent dataset from Lyft library
        randomize_frame (np.bool): Frames are yield in random order
        randomize_scene (np.bool): Scenes are yield in random order
        num_scenes (np.bool): Total number of scenes to be yield in one epoch
        frames_per_scene (np.bool): Number of frames to yield per scene (-1 means all)
        yield_only_large_distances (np.bool): Yield only samples with a given minimum maximum displacement.
        min_max_dist (np.float32): Minimum maximum displacement to use.
    """

    # Save the requested number of frames per scene
    requested_frames_per_scene = frames_per_scene

    # Scene order (incremental)
    scene_order = np.arange(num_scenes)
    # Shuffle scene order if requested
    if randomize_scene:
        np.random.shuffle(scene_order)

    # For each scene in the dataset
    for scene_idx in scene_order:

        # Frame order (incremental)
        this_scene_idxs = dataset.get_scene_indices(scene_idx)
        if this_scene_idxs.size < 1:
            continue
        # Shuffle frame order if requested
        if randomize_frame:
            np.random.shuffle(this_scene_idxs)

        # Get the number of frames to yield from this scene
        if requested_frames_per_scene > this_scene_idxs.shape[0]:
            frames_per_scene = this_scene_idxs.shape[0]
        elif requested_frames_per_scene<1:
            frames_per_scene = this_scene_idxs.shape[0]
        else:
            frames_per_scene = requested_frames_per_scene
        # Yield the number of requested frames of this scene
        for frame_idx in range(frames_per_scene):
            sample_index = this_scene_idxs[frame_idx]
            sample_out = dataset[sample_index]

            # Filter samples with low maximum displacement if requested
            if yield_only_large_distances:
                dist_max = np.sqrt(sample_out['target_positions'][np.sum(sample_out['target_availabilities']),0]**2\
                                  +sample_out['target_positions'][np.sum(sample_out['target_availabilities']),1]**2)
                if dist_max < min_max_dist:
                    continue

            # add index to sample
            sample_out['sample_idx'] = sample_index
            yield sample_out
    return

def get_tf_dataset(dataset,
                   num_hist_frames,
                   map_input_shape,
                   num_future_frames,
                   randomize_frame=False,
                   randomize_scene=False,
                   num_scenes=-1,
                   frames_per_scene = -1,
                   meta_dict_use = meta_dict_gen):
    """
    Creates a TensorFlow dict dataset from a Lyft dataset generator.

    Args:
        dataset (AgentDataset): Agent dataset from Lyft library
        num_hist_frames (np.int32): Number of history points in a frame
        map_input_shape (np.ndarray): Shape of the map image.
        num_future_frames (np.int32): Number of frames to be predicted.
        randomize_frame (np.bool): Frames are yield in random order
        randomize_scene (np.bool): Scenes are yield in random order
        num_scenes (np.int32): Total number of scenes to be yield in one epoch
        frames_per_scene (np.int32): Number of frames to yield per scene (-1 means all)
        meta_dict_use (function): Dictionary read function to be used, defaults to "meta_dict_gen".
    Returns:
        tf_dataset (tf.Dataset): TensorFlow Dataset object.
    """

    # Get number of scenes to use
    if num_scenes < 1:
        num_scenes = len(dataset.dataset.scenes)
    if num_scenes > len(dataset.dataset.scenes):
        print('Requested number of scenes (%d) exceeds the maximum, set to maximum: %d'%(num_scenes, len(dataset.dataset.scenes)))
        num_scenes = len(dataset.dataset.scenes)

    print('Creating dataset with: \n\t Randomized scenes: %r\n\t Randomized frames: %r\n\t Number of scenes: %d\n\t Number of frames per scenes: %d'%(randomize_frame, randomize_scene, num_scenes, frames_per_scene))

    # Create dict dataset and return
    return tf.data.Dataset.from_generator(lambda: meta_dict_use(dataset,
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
                                                        'extent': tf.float32,
                                                        'sample_idx': tf.int32},
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
                                                         'extent': (3,),
                                                         'sample_idx': ()})


def create_fade_image(input_multiCh_image, fade_factor = 0.75, update_threshold = 0.1):
    """
    Create an image with previous history layers faded in.

    Args:
        input_multiCh_image (np.ndarray): Ego or Agent map with channel size equal to number of history frames
        fade_factor (np.float32): Intensity fade of the history frames.
        update_threshold (np.float32): Minimum image activity to be kept.
    Returns:
        imgFade (np.ndarray): Ego or Agent map with faded history frames.
    """
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
def tf_get_input_sample(datasetSample, image_preprocess_fcn = lambda x: x, use_fading = True, use_angle = True):
    """
    Tensorflow sample mapping function.

    Takes a frame and pre-process its information, resulting in selected data to
    be used in the model.

    Args:
        datasetSample (dict) Sample tensors.
    Returns:
        sampleMapComp (tf.tensor): Multi-channel image with the RGB map and the faded Ego and Agents
        sampleHistPath (tf.tensor): Tensor of history [num_hist_frames x coordinates]
        sampleTargetPath (tf.tensor): Target tensor [num_objective_frames x coordinates]
        histAvail (tf.tensor): Bool tensor of history availability [num_hist_frames]
        targetAvail (tf.tensor): Bool tensor of target availability [num_objective_frames]
        timeStamp (tf.tensor): Time stamp of the sample
        trackID (tf.tensor): Track ID of the agent
        thisRasterFromAgent (tf.tensor): Raster from Agent conversion matrix.
        thisWorldFromAgent (tf.tensor): World from Agent conversion matrix.
        thisCentroid (tf.tensor): Agent centroid.
        thisSampleIdx (tf.tensor): Sample index.
    """

    # Get number of history frames
    num_hist_frames = datasetSample['history_positions'].shape[0]

    image_splits = tf.split(datasetSample['image'], num_or_size_splits=[num_hist_frames, num_hist_frames, 3], axis=0)

    # Map to RGB
    sampleMap = tf.transpose(image_splits[2], perm=[1, 2, 0])
    sampleMap *= 255
    sampleMap = image_preprocess_fcn(sampleMap)

    if use_fading:
        # Ego to fade
        sampleEgoFade = create_fade_image(image_splits[1])
        sampleEgoFade /= tf.reduce_max(sampleEgoFade)
        sampleEgoFade *= 255
        sampleEgoFade = tf.expand_dims(sampleEgoFade, axis=-1)
        sampleEgoFade = image_preprocess_fcn(sampleEgoFade)
    else:
        sampleEgoList = list()
        for i in range(num_hist_frames):
            sampleEgoList.append(image_preprocess_fcn(image_splits[1][i,:,:]))

    if use_fading:
        # Agents to fade
        sampleAgentsFade = create_fade_image(image_splits[0])
        sampleAgentsFade /= (tf.reduce_max(sampleAgentsFade)+1e-8) # Add a low value in case no other agent is present
        sampleAgentsFade *= 255
        sampleAgentsFade = tf.expand_dims(sampleAgentsFade, axis=-1)
        sampleAgentsFade = image_preprocess_fcn(sampleAgentsFade)
    else:
        sampleAgentsList = list()
        for i in range(num_hist_frames):
            sampleAgentsList.append(image_preprocess_fcn(image_splits[0][i,:,:]))

    # Concatenate image inputs
    if use_fading:
        sampleMapComp = tf.concat([sampleMap, sampleEgoFade, sampleAgentsFade], axis = -1)
    else:
        sampleEgoList = tf.convert_to_tensor(sampleEgoList)
        sampleEgoList = tf.reshape(sampleEgoList, shape=(image_splits[0].shape[1], image_splits[0].shape[2], -1))
        sampleAgentsList = tf.convert_to_tensor(sampleAgentsList)
        sampleAgentsList = tf.reshape(sampleAgentsList, shape=(image_splits[0].shape[1], image_splits[0].shape[2], -1))
        sampleMapComp = tf.concat([sampleMap, sampleEgoList, sampleAgentsList], axis = -1)


    # History path
    x = datasetSample['history_positions'][:,0]#/100.0
    x = tf.expand_dims(x, axis=-1)
    y = datasetSample['history_positions'][:,1]#/40.0
    y = tf.expand_dims(y, axis=-1)
    if use_angle:
        a = datasetSample['history_yaws']#/np.pi
        sampleHistPath = tf.concat([x, y, a], axis = -1)
    else:
        sampleHistPath = tf.concat([x, y], axis = -1)

    # Targets
    x = datasetSample['target_positions'][:,0]#/100.0
    x = tf.expand_dims(x, axis=-1)
    y = datasetSample['target_positions'][:,1]#/40.0
    y = tf.expand_dims(y, axis=-1)
    if use_angle:
        a = datasetSample['target_yaws']#/np.pi
        sampleTargetPath = tf.concat([x, y, a], axis = -1)
    else:
        sampleTargetPath = tf.concat([x, y], axis = -1)


    # Availability
    histAvail = datasetSample['history_availabilities']
    targetAvail = datasetSample['target_availabilities']

    # Data
    timeStamp = datasetSample['timestamp']
    trackID = datasetSample['track_id']
    thisRasterFromAgent = datasetSample["raster_from_agent"]
    thisWorldFromAgent = datasetSample["world_from_agent"]
    thisCentroid = datasetSample["centroid"]
    thisSampleIdx = datasetSample['sample_idx']

    return sampleMapComp, sampleHistPath, sampleTargetPath, histAvail, targetAvail, timeStamp, trackID, thisRasterFromAgent, thisWorldFromAgent, thisCentroid, thisSampleIdx




###############################################################################
# ------------------------ TF MODEL LOAD/SAVE ------------------------------- #
###############################################################################

def fill_defaults(cfg):
    """
    Loads the default values of the config.

    Args:
        cfg (dict): Read parameters.
    Returns:
        cfg (dict): Full parameter dict with defaults if some category is not defined.
    """

    base_cfg = load_config_data(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'default_config.yaml'))

    
    for key_1 in base_cfg:
        try:
            for key_2 in base_cfg[key_1]:
                if not (key_2 in cfg[key_1]):
                    cfg[key_1][key_2] = base_cfg[key_1][key_2]
        except:
            pass

    return cfg


def save_model(model_save, save_path, save_name, use_keras=True):
    """
    Save model using Keras API (single .h5 file) OR a HDF5 for weights and JSON for model.

    Args:
        model_save (keras.model): Keras model to save
        save_path (string): Output folder
        save_name (string): Name of the h5 file to be written
        use_keras (np.bool): Use keras API to save, else use JSON + HDF5
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

    Args:
        load_path (string): Model save folder
        load_name (string): Name of the h5 file to be read
        use_keras (np.bool): Use keras API to save, else use JSON + HDF5
        custom_obj_dict (dict): Dictionary of custom objects for keras API (i.e. custom layers)
    Returns:
        model (keras.model): TensorFlow Keras model.
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


def save_optimizer_state(optimizer, save_path, save_name):
    """
    Save keras.optimizers object state.

    Args:
        optimizer (keras.optimizer): Optimizer object.
        save_path (string): Path to save location.
        save_name (string): Name of the .npy file to be created.
    """

    # Create folder if it does not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save weights
    np.save(os.path.join(save_path, save_name), optimizer.get_weights())

    return

def load_optimizer_state(load_path, load_name, optimizer, model_train_vars):
    """
    Loads keras.optimizers object state.

    Args:
        load_path (string): Path to save location.
        load_name (string): Name of the .npy file to be read.
        optimizer (keras.optimizer): Optimizer object to be loaded.
        model_train_vars (list): List of model variables (obtained using Model.trainable_variables)
    """

    # Load optimizer weights
    opt_weights = np.load(os.path.join(load_path, load_name)+'.npy', allow_pickle=True)

    # dummy zero gradients
    zero_grads = [tf.zeros_like(w) for w in model_train_vars]
    # save current state of variables
    saved_vars = [tf.identity(w) for w in model_train_vars]

    # Apply gradients which don't do nothing with Adam
    optimizer.apply_gradients(zip(zero_grads, model_train_vars))

    # Reload variables
    [x.assign(y) for x,y in zip(model_train_vars, saved_vars)]

    # Set the weights of the optimizer
    optimizer.set_weights(opt_weights)


    return
