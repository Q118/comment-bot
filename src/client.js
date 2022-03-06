import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

console.log(tf.version)
// force the backend to use WASM
tf.setBackend('wasm');

tf.ready().then(() => { // run the code once the tf is available/'ready'
    console.log(tf.getBackend());
});