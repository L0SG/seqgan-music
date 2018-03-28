import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import cPickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import yaml
import shutil
import postprocessing as POST
import datetime
from tensorflow.python import debug as tf_debug
from pathos.multiprocessing import ProcessingPool as Pool

with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU']
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = config['EMB_DIM'] # embedding dimension
HIDDEN_DIM = config['HIDDEN_DIM'] # hidden state dimension of lstm cell
SEQ_LENGTH = config['SEQ_LENGTH'] # sequence length
START_TOKEN = config['START_TOKEN']
PRE_GEN_EPOCH = config['PRE_GEN_EPOCH'] # supervise (maximum likelihood estimation) epochs for generator
PRE_DIS_EPOCH = config['PRE_DIS_EPOCH'] # supervise (maximum likelihood estimation) epochs for discriminator
SEED = config['SEED']
BATCH_SIZE = config['BATCH_SIZE']
ROLLOUT_UPDATE_RATE = config['ROLLOUT_UPDATE_RATE']
GENERATOR_LR = config['generator_lr']
REWARD_GAMMA = config['reward_gamma']
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = config['dis_embedding_dim']
dis_filter_sizes = config['dis_filter_sizes']
dis_num_filters = config['dis_num_filters']
dis_dropout_keep_prob = config['dis_dropout_keep_prob']
dis_l2_reg_lambda = config['dis_l2_reg_lambda']
dis_batch_size = config['dis_batch_size']

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = config['TOTAL_BATCH']
# vocab size for our custom data
vocab_size = config['vocab_size']
# positive data, containing real music sequences
positive_file = config['positive_file']
# negative data from the generator, containing fake sequences
negative_file = config['negative_file']
valid_file = config['valid_file']
generated_num = config['generated_num']

epochs_generator = config['epochs_generator']
epochs_discriminator = config['epochs_discriminator']


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # unconditinally generate random samples
    # it is used for test sample generation & negative data generation
    # called per D learning phase

    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    # dump the pickle data
    with open(output_file, 'wb') as fp:
        cPickle.dump(generated_samples, fp, protocol=2)


def generate_samples_conditonal(sess, batch, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        start_token = batch[:, 0]
        prediction = trainable_model.predict(sess, batch, start_token)
        #prediction = np.argmax(prediction, axis=2)
        generated_samples.extend(prediction)
    with open(output_file, 'wb') as fp:
        cPickle.dump(generated_samples, fp, protocol=2)


def generate_samples_conditional_v2(sess, data_loader, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        data_loader.reset_pointer()
        random_batch = data_loader.next_batch()
        start_token = random_batch[:, 0]
        prediction = trainable_model.predict(sess, random_batch, start_token)
        generated_samples.extend(prediction)
    with open(output_file, 'wb') as fp:
        cPickle.dump(generated_samples, fp, protocol=2)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    # independent of D, the standard RNN learning
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch_condtional(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    # independent of D, the standard RNN learning
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        # provide start token instead of zero start
        trainable_model.start_token = tf.constant(batch[:, 0], dtype=tf.int32)
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)
    # reset the start token to zero
    trainable_model.start_token = tf.constant([START_TOKEN] * BATCH_SIZE, dtype=tf.int32)
    return np.mean(supervised_g_losses)

# new implementations
def calculate_train_loss_epoch(sess, trainableav_model, data_loader):
    # calculate the train loss for the generator
    # same for pre_train_epoch, but without the supervised grad update
    # used for observing overfitting and stability of the generator
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        # note the newly implementated method call for the model
        # calculate_nll_loss_step calculate the node up to g_loss, but does not calculate the update node
        g_loss = trainable_model.calculate_nll_loss_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def calculate_bleu(sess, trainable_model, data_loader):
    # bleu score implementation
    # used for performance evaluation for pre-training & adv. training
    # separate true dataset to the valid set
    # conditionally generate samples from the start token of the valid set
    # measure similarity with nltk corpus BLEU
    smoother = SmoothingFunction()

    data_loader.reset_pointer()
    bleu_avg = 0

    references = []
    hypotheses = []

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        # predict from the batch
        # TODO: which start tokens?
        start_tokens = batch[:, 0]
        #start_tokens = np.array([START_TOKEN] * BATCH_SIZE, dtype=np.int64)
        prediction = trainable_model.predict(sess, batch, start_tokens)
        # argmax to convert to vocab
        #prediction = np.argmax(prediction, axis=2)

        # cast batch and prediction to 2d list of strings
        batch_list = batch.astype(np.str).tolist()
        pred_list = prediction.astype(np.str).tolist()
        references.extend(batch_list)
        hypotheses.extend(pred_list)

    bleu = 0.
    # calculate bleu for each predicted seq
    # compare each predicted seq with the entire references
    # this is slow, use multiprocess
    def calc_sentence_bleu(hypothesis):
        return sentence_bleu(references, hypothesis, smoothing_function=smoother.method4)
    if __name__ == '__main__':
        p = Pool()
        result = (p.map(calc_sentence_bleu, hypotheses))
    bleu = np.mean(result)

    return bleu

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    # data loaders declaration
    # loaders for generator, discriminator, and additional validation data loader
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    eval_data_loader = Gen_Data_loader(BATCH_SIZE)

    # define generator and discriminator
    # general structures are same with the original model
    # learning rates for generator needs heavy tuning for general use
    # l2 reg for D & G also affects performance
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, GENERATOR_LR, REWARD_GAMMA)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # VRAM limitation for efficient deployment
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    # define saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    # generate real data from the true dataset
    gen_data_loader.create_batches(positive_file)
    # generate real validation data from true validation dataset
    eval_data_loader.create_batches(valid_file)

    time = str(datetime.datetime.now())[:-7]
    log = open('save/experiment-log-conditional' + str(time) + '.txt', 'w')
    log.write(str(config)+'\n')
    log.write('D loss: original\n')
    log.flush()

    #summary_writer = tf.summary.FileWriter('save/tensorboard/', graph=tf.get_default_graph())

    if config['pretrain'] == True:
        #  pre-train generator
        print 'Start pre-training...'
        log.write('pre-training...\n')
        for epoch in xrange(PRE_GEN_EPOCH):
            # calculate the loss by running an epoch
            loss = pre_train_epoch_condtional(sess, generator, gen_data_loader)

            # measure bleu score with the validation set
            bleu_score = calculate_bleu(sess, generator, eval_data_loader)

            # since the real data is the true data distribution, only evaluate the pretraining loss
            # note the absence of the oracle model which is meaningless for general use
            buffer = 'pre-train epoch: ' + str(epoch) + ' pretrain_loss: ' + str(loss) + ' bleu: ' + str(bleu_score)
            print(buffer)
            log.write(buffer + '\n')
            log.flush()

            # generate 5 test samples per epoch
            # it automatically samples from the generator and postprocess to midi file
            # midi files are saved to the pre-defined folder
            if epoch == 0:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                POST.main(negative_file, 5, str(-1)+'_vanilla_', 'midi_conditional')
            elif epoch == PRE_GEN_EPOCH - 1:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                POST.main(negative_file, 5, str(-PRE_GEN_EPOCH)+'_vanilla_', 'midi_conditional')


        print 'Start pre-training discriminator...'
        # Train 3 epoch on the generated data and do this for 50 times
        # this trick is also in spirit of the original work, but the epoch strategy needs tuning
        for epochs in range(PRE_DIS_EPOCH):
            generate_samples_conditional_v2(sess, gen_data_loader, generator, BATCH_SIZE, generated_num, negative_file)
            D_loss = 0
            for _ in range(3):
                dis_data_loader.load_train_data(positive_file, negative_file)
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
                    D_loss += discriminator.loss.eval(feed, session=sess)
            buffer = 'epoch: ' + str(epochs+1) + '  D loss: ' + str(D_loss/dis_data_loader.num_batch/3)
            print(buffer)
            log.write(buffer + '\n')
            log.flush()

        # save the pre-trained checkpoint for future use
        # if one wants adv. training only, comment out the pre-training section after the save
        save_checkpoint(sess, saver,PRE_GEN_EPOCH, PRE_DIS_EPOCH)

    # define rollout target object
    # the second parameter specifies target update rate
    # the higher rate makes rollout "conservative", with less update from the learned generator
    # we found that higher update rate stabilized learning, constraining divergence of the generator
    rollout = ROLLOUT(generator, ROLLOUT_UPDATE_RATE)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    if config['pretrain'] == False:
        # load checkpoint of pre-trained model
        load_checkpoint(sess, saver)

    # 0.001 to 0.01
    if config['x10adv_g'] == True:
        generator.learning_rate *= 10

    for total_batch in range(TOTAL_BATCH):
        G_loss = 0
        # Train the generator for one step
        for it in range(epochs_generator):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, config['rollout_num'], discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)
            G_loss += generator.g_loss.eval(feed, session=sess)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        D_loss = 0
        for _ in range(epochs_discriminator):
            generate_samples_conditional_v2(sess, gen_data_loader, generator, BATCH_SIZE, generated_num, negative_file)
            for _ in range(config['epochs_discriminator_multiplier']):
                dis_data_loader.load_train_data(positive_file, negative_file)
                dis_data_loader.reset_pointer()

                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
                    D_loss += discriminator.loss.eval(feed, session=sess)

        # measure stability and performance evaluation with bleu score
        bleu_score = calculate_bleu(sess, generator, eval_data_loader)
        buffer = 'epoch: ' + str(total_batch + 1) + \
                 ',  G_adv_loss: %.12f' % (G_loss / epochs_generator) + \
                 ',  D loss: %.12f' % (D_loss / epochs_discriminator / config['epochs_discriminator_multiplier']) + \
                 ',  bleu score: %.12f' % bleu_score
        print(buffer)
        log.write(buffer + '\n')
        log.flush()

        if config['infinite_loop'] is True:
            if bleu_score < config['loop_threshold']:
                buffer = 'Mode collapse detected, restarting from pretrained model...'
                print(buffer)
                log.write(buffer + '\n')
                log.flush()
                load_checkpoint(sess, saver)

        # generate random test samples and postprocess the sequence to midi file
        #generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)

        # instead of the above, generate samples conditionally
        # randomly sample a batch
        # rng = np.random.randint(0, high=gen_data_loader.num_batch, size=1)
        # random_batch = np.squeeze(gen_data_loader.sequence_batch[rng])
        generate_samples_conditional_v2(sess, gen_data_loader, generator, BATCH_SIZE, generated_num, negative_file)
        POST.main(negative_file, 5, str(total_batch)+'_vanilla_', 'midi_conditional')
    log.close()


# methods for loading and saving checkpoints of the model
def load_checkpoint(sess, saver):
    #ckpt = tf.train.get_checkpoint_state('save')
    #if ckpt and ckpt.model_checkpoint_path:
    #saver.restore(sess, tf.train.latest_checkpoint('save'))
    ckpt = 'pretrain_g'+str(config['PRE_GEN_EPOCH'])+'_d'+str(config['PRE_DIS_EPOCH'])+'_vanilla_conditional.ckpt'
    saver.restore(sess, './save/' + ckpt)
    print 'checkpoint {} loaded'.format(ckpt)
    return


def save_checkpoint(sess, saver, g_ep, d_ep):
    checkpoint_path = os.path.join('save', 'pretrain_g'+str(g_ep)+'_d'+str(d_ep)+'_vanilla_conditional.ckpt')
    saver.save(sess, checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    return

if __name__ == '__main__':
    main()