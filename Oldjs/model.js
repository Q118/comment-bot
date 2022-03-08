import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


export const train = () => {
    tf.tidy(() => {
        run();
    });
};

const csvUrl = 'data/toxic_data_sample.csv';

const stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

let tmpDictionary = {};
let EMBEDDING_SIZE = 1000;
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

//fucntion to perform the tokenization process
const tokenize = (sentence, isCreateDict = false) => {
    const tmpTokens = sentence.split(/\s+/g); // split on white-spaces

    const tokens = tmpTokens.filter((token) => { //filtering out empty tokens and stop-words
        !stopWords.includes(token) && token.length > 0
    });
    if (isCreateDict) {
        const labelCounts = tokens.reduce((acc, token) => {
            acc[token] = acc[token] === undefined ? 1 : acc[token] += 1;
            return acc;
        }, tmpDictionary);
    }
    return tmpTokens;
}

const run = async () => {
    const rawDataResult = readRawData();
    const labels = [];

    const comments = [];

    const documentTokens = []; // store the tokens for each document

    await rawDataResult.forEachAsync(row => {
        // seperate out our comments into words
        const comment = row['xs']['comment_text'];
        const trimmedComment = comment.toLowerCase().trim();
        comments.push(trimmedComment);

        documentTokens.push(tokenize(trimmedComment, true));

        labels.push(row['ys']['toxic']);
    });
    // plot labels
    plotOutputLabelCounts(labels);

    console.log(Object.keys(tmpDictionary).length);
    console.log(tmpDictionary);


}