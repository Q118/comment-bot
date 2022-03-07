import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


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

const plotOutputLabelCounts = (labels) => {
    const labelCounts = labels.reduce((acc, label) => {
        // take each label and check if we have seen it previously
        acc[label] = acc[label] === undefined ? 1 : acc[label] + 1;
        // increment if have, if not then create a key with a counter set to 1
        return acc;
    }, {});
    // console.log(labelCounts);
    const barChartData = [];

    Object.keys(labelCounts).forEach((key) => {
        barChartData.push({
            index: key,
            value: labelCounts[key]
        });
    });
    //    console.log(barChartData)
    tfvis.render.barchart({
        tab: 'Exploration',
        name: 'Toxic output labels'
    }, barChartData);




}


const run = async () => {
    const rawDataResult = readRawData();
    const labels = [];

    await rawDataResult.forEachAsync(row => {
        // console.log(row);
        labels.push(row['ys']['toxic']);
    });
    // plot labels
    plotOutputLabelCounts(labels);
}