import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

export const train = () => {
    tf.tidy(() => {
        run();
    });
    
};

const csvUrl = 'data/toxic_data_sample.csv';
const stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
let tmpDictionary = {};
let EMBEDDING_SIZE = 1000;
const BATCH_SIZE = 16;

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

const plotOutputLabelCounts = (labels) => {
    const labelCounts = labels.reduce((acc, label) => {
        acc[label] = acc[label] === undefined ? 1 : acc[label] +=1;
        return acc;
    },{});

    //console.log(labelCounts);
    const barChartData = [];
    Object.keys(labelCounts).forEach((key) => {
        barChartData.push({
            index: key,
            value: labelCounts[key]
        });
    });
    //console.log(barChartData);
    tfvis.render.barchart({
        tab: 'Exploration',
        name: 'Toxic output labels'
    }, barChartData);
}

const tokenize = (sentence, isCreateDict = false) => {
    const tmpTokens = sentence.split(/\s+/g);
    const tokens = tmpTokens.filter((token) => !stopwords.includes(token) && token.length > 0);

    if (isCreateDict) {
        const labelCounts = tokens.reduce((acc, token) => {
                acc[token] = acc[token] === undefined ? 1 : acc[token] +=1;
            return acc;
        },tmpDictionary);
    }
    return tmpTokens;
}

const sortDictionaryByValue = (dict) => {
    const items = Object.keys(dict).map((key) => {
        return [key, dict[key]];
    });
    return items.sort((first, second) => {
        return second[1] - first[1];
    });
}

const getInverseDocumentFrequency = (documentTokens, dictionary) => {
    return dictionary.map((token) => 1 + Math.log(documentTokens.length / documentTokens.reduce((acc, curr) => curr.includes(token) ? acc + 1 : acc, 0)))
}

const encoder = (sentence, dictionary, idfs) => {
    const tokens = tokenize(sentence);
    const tfs = getTermFrequency(tokens, dictionary);
    const tfidfs = getTfIdf(tfs, idfs);
    return tfidfs;
}

const getTermFrequency = (tokens, dictionary) => {
    return dictionary.map((token) => tokens.reduce((acc, curr) => curr == token ? acc + 1 : acc, 0))
}

const getTfIdf = (tfs, idfs) => {
    return tfs.map((element, index) => element * idfs[index])
}




const prepareData = (dictionary, idfs) => {

    const preprocess = ({ xs, ys }) => {
        const comment = xs['comment_text'];
        const trimedComment = comment.toLowerCase().trim();
        const encoded = encoder(trimedComment, dictionary, idfs);

        return {
            xs: tf.tensor2d([encoded], [1, dictionary.length]), 
            ys: tf.tensor2d([ys['toxic']], [1, 1]) 
        }

    }

    // load data
    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    })
    .map(preprocess);

    return readData;
}

const prepareDataUsingGenerator = (comments, labels, dictionary, idfs) => {
    function* getFeatures() {
        for (let i = 0; i < comments.length; i++) {
            // Generate one sample at a time.
            const encoded = encoder(comments[i], dictionary, idfs);
            yield tf.tensor2d([encoded], [1, dictionary.length]);
        }
    }
    function* getLabels() {
        for (let i = 0; i < labels.length; i++) {
            yield tf.tensor2d([labels[i]], [1, 1]);
        }
    }

    const xs = tf.data.generator(getFeatures);
    const ys = tf.data.generator(getLabels);
    const ds = tf.data.zip({ xs, ys });
    return ds;

}

const trainValTestSplit = (ds, nrows) => {
    // Train Test Split 
    const trainingValidationCount = Math.round(nrows * 0.7);
    const trainingCount = Math.round(nrows * 0.6);

    const SEED = 7687547;

    const trainingValidationData =
        ds
            .shuffle(nrows, SEED)
            .take(trainingValidationCount);

    const testDataset =
        ds
            .shuffle(nrows, SEED)
            .skip(trainingValidationCount)
            .batch(BATCH_SIZE);

    const trainingDataset =
        trainingValidationData
            .take(trainingCount)
            .batch(BATCH_SIZE);

    const validationDataset =
        trainingValidationData
            .skip(trainingCount)
            .batch(BATCH_SIZE);

    // return values
    return {
        trainingDataset,
        validationDataset,
        testDataset
    };
}


const run = async () => {
    const rawDataResult = readRawData();
    const labels = [];

    const comments = [];
    const documentTokens = [];
    await rawDataResult.forEachAsync((row) => {
        //console.log(row);
        const comment = row['xs']['comment_text'];
        const trimedComment = comment.toLowerCase().trim();
        comments.push(trimedComment);
        documentTokens.push(tokenize(trimedComment, true));
        labels.push(row['ys']['toxic']);
    });

    // plot labels
    plotOutputLabelCounts(labels);

    console.log(Object.keys(tmpDictionary).length);
    //console.log(tmpDictionary);

    const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
    if (sortedTmpDictionary.length  <= EMBEDDING_SIZE){
        EMBEDDING_SIZE = sortedTmpDictionary.length;
    }

    //console.log(sortedTmpDictionary);
    const dictionary = sortedTmpDictionary.slice(0, EMBEDDING_SIZE).map((row) => row[0]);

    // console.log('dictionary length ' + dictionary.length);
    // console.log(dictionary);

    // calculate IDF
    const idfs = getInverseDocumentFrequency(documentTokens, dictionary);
    //console.log(idfs);

    // Processing data 
    const ds = prepareData(dictionary, idfs);
    // await ds.forEachAsync((e) => console.log(e));

    //Sample test code
    // const documentTokens = [];
    // const testComments = ['i loved the movie', 'movie was boring'];
    // testComments.forEach((row) => {
    //     const comment = row.toLowerCase();
    //     documentTokens.push(tokenize(comment, true));
    // });

    // console.log(tmpDictionary);
    // const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
    // const dictionary = sortedTmpDictionary.map((row) => row[0]);
    // const idfs = getInverseDocumentFrequency(documentTokens, dictionary);
 

    // testComments.forEach((row) => {
    //     const comment = row.toLowerCase();       
    //     console.log(encoder(comment,dictionary, idfs));
    // });

    // using generator
    // const ds = prepareDataUsingGenerator(comments, labels, dictionary, idfs)
    //await ds.forEachAsync((e) => console.log(e));

    const { trainingDataset, validationDataset, testDataset } = trainValTestSplit(ds, documentTokens.length);
    await trainingDataset.forEachAsync((e) => console.log(e));
}