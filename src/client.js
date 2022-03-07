import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import "regenerator-runtime/runtime";
// import * as model from './model';

// console.log(tf.version);
// tf.setBackend('wasm');
// tf.ready().then(() => {
//     console.log(tf.getBackend());
// });

// //tensor1d
// const age = tf.tensor1d([30, 25],'int32');
// age.print();
// tf.print(age);
// console.log(age.shape);
// console.log(age.dtype);

// // //tensor2d
// const age_income_height = tf.tensor2d([[30, 1000, 170], [25, 2000, 168]]);
// age_income_height.print();
// console.log(age_income_height.shape);
// console.log(age_income_height.dtype);

// //scalar
// const multiplier = tf.scalar(10);
// multiplier.print();
// console.log(multiplier.dtype);

// const y = tf.tidy(() => {
// // //Tensor Operations
// // //addition
// const income_source_1 = tf.tensor1d([100, 200, 300, 150]);
// const income_source_2 = tf.tensor1d([50, 70, 30, 20]);
// // const total_income = tf.add(income_source_1, income_source_2);
// const total_income = tf.keep( income_source_1.add(income_source_2));
// total_income.print();

// //variables
// const var_1 = tf.variable(income_source_1);
// tf.print('var_1 before assignment : ' + var_1);
// var_1.assign(income_source_2);
// tf.print('var_1 after assignment : ' + var_1);

// // var_1.dispose();
// // total_income.dispose();
// console.log(tf.memory().numTensors);
// });

// console.log('Number of tensors after clean up : ' + tf.memory().numTensors);


console.log("line1");
console.log("line2");
const init = async () => {
    await tf.ready();
    console.log(tf.getBackend());
    // model.train();
}
init();
console.log("line4");