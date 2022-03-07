import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';


// console.log(tf.version)
// // force the backend to use WASM
// tf.setBackend('wasm');
// tf.ready().then(() => { // run the code once the tf is available/'ready'
//     console.log(tf.getBackend());
// });

//tensor1d
const age = tf.tensor1d([30, 25], 'int32');
age.print();
tf.print(age);
console.log(age.shape);
console.log(age.dtype);

//tesnsor2d
const age_income_height = tf.tensor2d([
    [30, 1000, 170], [25, 2000, 168]
]);
age_income_height.print();
console.log(age_income_height.shape);
console.log(age_income_height.dtype);

//scalar
const multiplier = tf.scalar(10);
multiplier.print();
console.log(multiplier.dtype);

// tesors are immutablle.. i.e. any operation performed on them create a new tensor






import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

// Tensor Operations
// addition
tf.tidy(() => {
    const income_source_1 = tf.tensor1d([100, 200, 300, 150]);
    const income_source_2 = tf.tensor1d([50, 70, 30, 20]);

    // const total_income = tf.add(income_source_1, income_source_2);
    const total_income = income_source_1.add(income_source_2);
    total_income.print();


    /*
    * Sometimes you might want to create variables
    * that can store temporary values that get updated
    *  during some process, such as model training.
    * In such case, you can create variables.
    */
    // variables (use them kind of like pointers!)
    const var_1 = tf.variable(income_source_1);
    tf.print(`var_1 before assignment: ${var_1}`);
    var_1.assign(income_source_2);
    tf.print(`var_1 after assignment: ${var_1}`);


    //var_1.dispose();

    // memory management
    console.log(tf.memory().numTensors);
    // solve with tidy
    // tf.tidy(() => {
    //     const x = tf.tensor1d([1, 2, 3]); // gets removed
    //     const y = tf.scalar(10); // gets removed
    //     z = tf.keep(x.square); // keep this tensor in memory
    // });

});

console.log(`Number of tensors after Tidy-Up: ${tf.memory().numTensors}`);