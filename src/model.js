import * as tf from '@tensorflow/tfjs';

const readRawData = () => {
    // load data
    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    });
    return readData;
}

const run = () => {
}