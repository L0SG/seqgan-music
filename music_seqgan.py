import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import cPickle
import os
from nltk.translate.bleu_score import corpus_bleu
import yaml
import shutil

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
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    # dump the pickle data
    with open(negative_file, 'wb') as fp:
        cPickle.dump(generated_samples, fp, protocol=2)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

# new implementation
def calculate_train_loss_epoch(sess, trainable_model, data_loader):
    # calculate the train loss for the generator
    # same for pre_train_epoch, but without the supervised grad update
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = trainable_model.calculate_nll_loss_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def calculate_bleu(sess, trainable_model, data_loader):
    data_loader.reset_pointer()
    bleu_avg = 0

    for it in xrange(data_loader.num_batch):
        batch =data_loader.next_batch()
        # predict from the batch
        start_tokens = batch[:, 0]
        prediction = trainable_model.predict(sess, batch, start_tokens)
        # argmax to convert to vocab
        prediction = np.argmax(prediction, axis=2)

        # cast batch and prediction to 2d list of strings
        batch_list = batch.astype(np.str).tolist()
        pred_list = prediction.astype(np.str).tolist()

        bleu = 0
        # calculate bleu for each sequence
        for i in range(len(batch_list)):
            bleu += corpus_bleu(batch_list[i], pred_list[i])
        bleu = bleu / len(batch_list)
        bleu_avg += bleu
    bleu_avg = bleu_avg / data_loader.num_batch

    return bleu_avg

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    # assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    eval_data_loader = Gen_Data_loader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # generate real data from the true dataset
    gen_data_loader.create_batches(positive_file)
    # generate real validation data from true validation dataset
    eval_data_loader.create_batches(valid_file)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_GEN_EPOCH):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        bleu_score = calculate_bleu(sess, generator, eval_data_loader)
        # since the real data is the true data distribution, only evaluate the pretraining loss
        if epoch % 1 == 0:
            buffer = 'pre-train epoch: '+ str(epoch+1) + '  pretrain_loss: '+ str(loss) + '  bleu: '+ str(bleu_score)
            print(buffer)
            log.write(buffer)

    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for epochs in range(PRE_DIS_EPOCH):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        D_loss = 0
        for _ in range(3):
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
        log.write(buffer)
    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        G_loss = 0
        # Train the generator for one step
        for it in range(epochs_generator):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)
            G_loss += generator.g_loss.eval(feed, session=sess)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        D_loss = 0
        for _ in range(epochs_discriminator):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
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
        buffer = 'epoch: ' + str(total_batch+1) + \
                 ',  G_adv_loss: %.12f' % (G_loss/epochs_generator) + \
                 ',  D loss: %.12f' % (D_loss/epochs_discriminator/3) + \
                 ',  bleu score: %.12f' % calculate_bleu(sess, generator, eval_data_loader)
        print(buffer)
        log.write(buffer)
        # save first generated samples
        if (total_batch+1) == 1:
            shutil.copy2(negative_file, './save/gen_epoch_'+str(total_batch+1))
            print 'gen_epoch_'+str(total_batch+1)+' data saved!'
        # save every 10 epoch, save generated music
        if (total_batch+1) % 10 == 0:
            shutil.copy2(negative_file, './save/gen_epoch_'+str(total_batch+1))
            print 'gen_epoch_' + str(total_batch + 1) + ' data saved!'
    log.close()


if __name__ == '__main__':
    main()