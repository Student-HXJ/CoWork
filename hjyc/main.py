import os
# import pdb
import time

import math
from PIL import Image
from keras import Model, Input
from keras.engine.saving import load_model
from keras.layers import LSTM, Add, Dense, Subtract, Concatenate, Dropout, \
    Conv2D, MaxPooling2D, TimeDistributed, ConvLSTM2D, BatchNormalization, \
    AveragePooling2D, CuDNNLSTM, Reshape, RepeatVector, LeakyReLU, \
    Conv2DTranspose, Activation
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from attention_decoder_ad import AttentionDecoder
from loss_history import LossHistory

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class Custom_Model():
    steps = 12

    # not used
    def dense(self):
        units = 16
        input = Input(shape=(self.steps, 50, 50), name='input')
        x = TimeDistributed(Dense(units))(input)
        x = Activation(activation='relu')(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(Dense(units))(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(Dense(50))(x)
        model = Model(inputs=input, outputs=x)
        return model

    def seq2seq(self):
        units = 128
        input = Input(shape=(self.steps, 50, 50), name='input')
        x = Reshape((self.steps, 50, 50, 1))(input)
        x = TimeDistributed(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=True))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(Conv2D(filters=4, kernel_size=3, strides=1, padding='same', use_bias=True))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, padding='same'))(x)
        x = Reshape((self.steps, -1))(x)
        x = TimeDistributed(Dense(units))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        # encoder
        x, state_h, state_c = CuDNNLSTM(units, return_state=True, return_sequences=True)(x)
        state = [state_h, state_c]
        x = BatchNormalization()(x)
        '''
        # decoder
        x = Dense(units)(x)
        x = Activation(activation='relu')(x)
        x = RepeatVector(self.steps)(x)
        x = Dropout(0.05)(x)
        x = BatchNormalization()(x)
        '''

        x = AttentionDecoder(units=units, output_dim=256)(x)
        # decoder = CuDNNLSTM(units, return_sequences=True)
        # x = decoder(x, initial_state=state)
        # x = decoder(x)

        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(Dense(50 * 50))(x)
        x = LeakyReLU(alpha=0.05)(x)
        # x = TimeDistributed(Dense(50 * 50))(x)
        x = Reshape((-1, 50, 50))(x)

        # input = Input(shape=(self.steps, self.height, self.width, 1,),
        #               name='input')
        # x = ConvLSTM2D(filters=10,
        #                kernel_size=3,
        #                strides=1,
        #                padding='same',
        #                use_bias=True,
        #                return_sequences=True,
        #                data_format='channels_last')(input)
        # x = ConvLSTM2D(filters=10,
        #                kernel_size=3,
        #                strides=1,
        #                padding='same',
        #                use_bias=True,
        #                return_sequences=True,
        #                data_format='channels_last')(x)
        # x = ConvLSTM2D(filters=10,
        #                kernel_size=3,
        #                strides=1,
        #                padding='same',
        #                use_bias=True,
        #                return_sequences=False,
        #                data_format='channels_last')(x)
        # x = Dense(1, activation='linear')(x)

        model = Model(inputs=input, outputs=x)
        return model

    def res(self):
        units = 128
        in_vc = Input(shape=(self.steps, 50, 50))
        in_ec = Input(shape=(self.steps, 50, 50))
        vc = Reshape((self.steps, 50, 50, 1))(in_vc)
        ec = Reshape((self.steps, 50, 50, 1))(in_ec)
        x = Concatenate(axis=-1)([vc, ec])

        x = TimeDistributed(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=True))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(Conv2D(filters=4, kernel_size=3, strides=1, padding='same', use_bias=True))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, padding='same'))(x)
        x = Reshape((self.steps, -1))(x)
        x = TimeDistributed(Dense(units))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        # encoder
        x, state_h, state_c = CuDNNLSTM(units, return_state=True, return_sequences=False)(x)
        state = [state_h, state_c]
        x = BatchNormalization()(x)

        # decoder
        x = Dense(units)(x)
        x = Activation(activation='relu')(x)
        x = RepeatVector(self.steps)(x)
        x = Dropout(0.05)(x)
        x = BatchNormalization()(x)

        # x = AttentionDecoder(units=units, output_dim=256)(x)
        decoder = CuDNNLSTM(units, return_sequences=True)
        x = decoder(x, initial_state=state)
        # x = decoder(x)

        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)
        x = TimeDistributed(Dense(50 * 50))(x)
        x = LeakyReLU(alpha=0.05)(x)
        # x = TimeDistributed(Dense(50 * 50))(x)
        x = Reshape((-1, 50, 50))(x)

        model = Model(inputs=[in_vc, in_ec], outputs=x)
        return model

    def build(self):
        vp = Input(shape=(self.steps, 50, 50), name='V_p')
        vc = Input(shape=(self.steps, 50, 50), name='V_c')

        seq2seq = self.seq2seq()
        seq2seq.name = 'Seq2Seq'
        res = self.res()
        res.name = 'Res'
        print(seq2seq.summary())
        print(res.summary())

        vc_e = seq2seq(vp)
        ec = Subtract()([vc, vc_e])

        # con = Concatenate()()
        ef_ = res([vc, ec])

        vf_e = seq2seq(vc)
        add = Add()([vf_e, ef_])
        vf_ = Model(inputs=[vp, vc], outputs=add, name='Model')
        print(vf_.summary())
        return vf_


def create_scaler(folder):
    files = []
    for e in os.listdir(folder):
        files.append(e)
    img = Image.open(os.path.join(folder, files[0]))
    img = np.array(img)
    img[img < 0] = 0
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(img.reshape(-1, 1))
    return scaler


def img_generator(folder, scaler, steps, batch_size=1, contain_previous=False):

    files = []
    for e in os.listdir(folder):
        files.append(e)
    while True:
        batch_p, batch_c, batch_f = [], [], []
        for sample_index in range(batch_size):

            if not contain_previous:
                state = int(np.random.randint(0, len(files) - 2 * steps, 1))
            else:
                state = int(np.random.randint(0, len(files) - 3 * steps, 1))

            # 整个图太大了。。。拿个小一点的 50 * 50 做个实验
            pos_w = int(np.random.randint(0, math.ceil(500 / 50), 1)) * 50
            pos_h = int(np.random.randint(0, math.ceil(300 / 50), 1)) * 50

            pos_w = 250
            pos_h = 250

            vp, vc, vf = [], [], []
            for n in range(steps):
                img = Image.open(os.path.join(folder, files[state + n]))
                img = img.crop((pos_w, pos_h, pos_w + 50, pos_h + 50))
                img = np.array(img)
                img[img < 0] = 0
                vp.append(img)
            for n in range(steps, 2 * steps):
                img = Image.open(os.path.join(folder, files[state + n]))
                img = img.crop((pos_w, pos_h, pos_w + 50, pos_h + 50))
                img = np.array(img)
                img[img < 0] = 0
                vc.append(img)
            if contain_previous:
                for n in range(2 * steps, 3 * steps):
                    img = Image.open(os.path.join(folder, files[state + n]))
                    img = img.crop((pos_w, pos_h, pos_w + 50, pos_h + 50))
                    img = np.array(img)
                    img[img < 0] = 0
                    vf.append(img)
            batch_p.append(np.asarray(vp))
            batch_c.append(np.asarray(vc))
            batch_f.append(np.asarray(vf))
        batch_p = transform(scaler, np.asarray(batch_p))
        batch_c = transform(scaler, np.asarray(batch_c))
        if not contain_previous:
            yield (batch_p, batch_c)
        else:
            batch_f = transform(scaler, np.asarray(batch_f))
            yield ([batch_p, batch_c], batch_f)


def transform(scaler, data):
    shape = data.shape
    return scaler.transform(data.reshape(-1, 1)).reshape(shape)


def inv_transform(scaler, data):
    shape = data.shape
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(shape)


# rubbish
def test_generator(steps):
    folder = 'China_PM25'
    folder_test = 'China_PM2.5_T'
    files = []
    for e in os.listdir(folder):
        files.append(e)

    state = len(files) - 2 * steps
    # pos_w = int(
    #     np.random.randint(0, math.ceil(self.width / 50), 1)) * 50
    # pos_h = int(
    #     np.random.randint(0, math.ceil(self.height / 50), 1)) * 50

    pos_w = 250
    pos_h = 250

    vp, vc, vf = [], [], []
    for n in range(steps):
        img = Image.open(os.path.join(folder, files[state + n]))
        img = img.crop((pos_w, pos_h, pos_w + 50, pos_h + 50))
        img = np.array(img)
        img[img < 0] = 0
        vp.append(img)
    for n in range(steps, 2 * steps):
        img = Image.open(os.path.join(folder, files[state + n]))
        img = img.crop((pos_w, pos_h, pos_w + 50, pos_h + 50))
        img = np.array(img)
        img[img < 0] = 0
        vc.append(img)

    files = []
    for e in os.listdir(folder_test):
        files.append(e)
    for n in range(steps):
        img = Image.open(os.path.join(folder_test, files[n]))
        img = img.crop((pos_w, pos_h, pos_w + 50, pos_h + 50))
        img = np.array(img)
        img[img < 0] = 0
        vf.append(img)

    vp = transform(scaler, np.asarray([vp]))
    vc = transform(scaler, np.asarray([vc]))
    vf = transform(scaler, np.asarray([vf]))
    return [vp, vc], vf


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('start.', time_str())
    c_model = Custom_Model()
    model = c_model.build()

    model.compile(optimizer='adam', loss='mse')
    # plot_model(model, to_file='custom_model.png', show_shapes=True)
    scaler = create_scaler(folder='China_PM25')

    # pdb.set_trace()
    # 整图卷积搞不定，部分卷。。。50 * 50

    img_generator = img_generator(folder='China_PM25', scaler=scaler, steps=12, batch_size=5, contain_previous=True)
    history = LossHistory()
    model.fit_generator(img_generator, epochs=200, samples_per_epoch=5, verbose=2, callbacks=[history])
    history.loss_plot('epoch', savepath='plot2.png')
    print(history.losses['epoch'])

    # pdb.set_trace()

    source = test_generator(steps=12)
    prediction = model.predict(source[0])
    actual = source[1]
    prediction = inv_transform(scaler, prediction).reshape(12, 50, 50)
    actual = inv_transform(scaler, actual).reshape(12, 50, 50)
    prediction[prediction < 0] = 0

    mse = np.mean(np.square(prediction.flatten() - actual.flatten()))
    rmse = np.math.sqrt(mse)
    mae = np.mean(np.abs(prediction.flatten() - actual.flatten()))
    r2 = 1 - mse / np.var(actual)
    print('MSE=%.3f\tRMSE=%.3f\tMAE=%.3f\tR2=%.3f' % (mse, rmse, mae, r2))

    if not os.path.exists('out'):
        os.makedirs('out')

    for n_step in range(c_model.steps):
        array_predict = prediction[n_step, :, :]
        array_actual = actual[n_step, :, :]

        img1 = Image.fromarray(array_predict)
        img1.save(os.path.join('out', 'predict_' + str(n_step) + '.tif'))
        img2 = Image.fromarray(array_actual)
        img2.save(os.path.join('out', 'label_' + str(n_step) + '.tif'))

        mse = np.mean(np.square(array_predict.flatten() - array_actual.flatten()))
        rmse = np.math.sqrt(mse)
        mae = np.mean(np.abs(array_predict.flatten() - array_actual.flatten()))
        r2 = 1 - mse / np.var(array_actual)
        print('%.3f\t%.3f\t%.3f\t%.3f' % (mse, rmse, mae, r2))
    # pdb.set_trace()
