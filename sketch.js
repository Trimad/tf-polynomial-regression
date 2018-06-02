let x_arr = [];
let y_arr = [];

let coefficient = [];

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);


function setup() {
  createCanvas(windowWidth, windowHeight);
  initializeConstants(2);
}

function initializeConstants(degree) {
  for (let i = 0; i <= degree; i++) {
    coefficient[i] = tf.variable(tf.scalar(null));
    console.log(coefficient[i]);
  }
}

function getYs(degreesLeft) {
//this is where I left off

}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}
function predict(x) {
  const xs = tf.tensor1d(x);
  const ys = xs.pow(tf.scalar(2)).mul(coefficient[0])
    .add(xs.mul(coefficient[1]))
    .add(coefficient[2]);
  /*
  const ys = sx.pow(tf.scalar(3)).mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
*/

  return ys;

}
function mousePressed() {
  const x = map(mouseX, 0, width, -1, 1);
  const y = map(mouseY, 0, height, -1, 1);
  x_arr.push(x);
  y_arr.push(y);
}
function draw() {

  drawGraph();
  if (x_arr.length > 0) {
    tf.tidy(function () {
      const ys = tf.tensor1d(y_arr);
      optimizer.minimize(() => loss(predict(x_arr), ys));
    }
    );
  }

  stroke(255);
  strokeWeight(8);

  //Draw every point
  for (let i = 0; i < x_arr.length; i++) {
    point(map(x_arr[i], -1, 1, 0, width), map(y_arr[i], -1, 1, 0, height));
  }

  const curveX = [];

  //The more points, the smoother the curve that is drawn/
  for (let x = -1; x <= 1.01; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();
  beginShape();
  strokeWeight(2);
  noFill();
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, 0, height);
    vertex(x, y);
  }
  endShape();
  if (frameCount % 15 === 0) {
    document.title = "FPS: " + Math.round(frameRate()) + " Tensors: " + tf.memory().numTensors;
    //console.log(a + ", " + b + ", " + c);
  }
}

function drawGraph() {
  background(0);
  for (var i = 0; i < height; i += 5) {
    stroke(51);
    point(width / 2, i);
  }
  for (var i = 0; i < width; i += 5) {
    stroke(51);
    point(i, height / 2);
  }

}