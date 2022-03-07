import * as tf from '@tensorflow/tfjs';

export const train = () => {
    tf.tidy(() => {
        run();
    });
};

const csvUrl = 'data/toxic_data_sample.csv';
const readRawData = () => {
    // load data
    const readData = tf.data.csv(csvUrl, {
        // Inside this object, you can specify if you want any specific column to be treated as the output label. 
        columnConfigs: {
            toxic: {
                isLabel: true // here we are marking the toxic column as the output label by setting the isLabel property to true.
            }
        }
    });
    return readData;
}

const run = async () => {
    const rawDataResult = readRawData();

    await rawDataResult.forEachAsync(row => {
        console.log(row);
    });

}