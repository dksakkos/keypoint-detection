import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.models import load_model

from data_ops import get_data, split_data, expand_to_3_channels
from generators import batch_gen, gen

sm.set_framework('tf.keras')
sm.framework()
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


chk = ModelCheckpoint('model.h5', monitor='val_f1-score', verbose=1, save_best_only=True,save_weights_only=False, mode='max', period=1)
redu = ReduceLROnPlateau(monitor='val_f1-score', factor=0.1, patience=3, min_lr=1e-7, verbose=1, mode='max')
early = EarlyStopping(monitor='val_f1-score', min_delta=1e-4, patience=10, verbose=0, mode='max')
csv_logger = CSVLogger('log.csv', append=True, separator=',')


model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    optimizer=Adam(learning_rate=0.002),
    loss=sm.losses.dice_loss,
    metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
)

images, labels = get_data()
images = preprocess_input(images)
train, val, test = split_data(images, labels)

model.fit(batch_gen(gen(train[0], train[1], augment=True), 32),
   epochs=100,
   callbacks=[chk, redu, early, csv_logger],
   validation_data=(expand_to_3_channels(val[0]), val[1]),
   steps_per_epoch=50
)

model = load_model('model.h5', custom_objects={'dice_loss': sm.losses.dice_loss,
                                               'iou_score': sm.metrics.iou_score,
                                               'f1-score': sm.metrics.f1_score})

scores = model.evaluate(expand_to_3_channels(test[0]), test[1])
for metric, score in zip (scores, model.metrics_names):
    print(score, '=', metric)
