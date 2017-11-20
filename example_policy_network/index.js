"use strict";

document.getElementById("debug").innerHTML = 'Loading network... (takes 5~30 seconds)';
var request = new XMLHttpRequest();
request.open('GET', "model-39540.net", true);
request.responseType = 'blob';
request.onload = function () {
    var reader = new FileReader();
    reader.readAsArrayBuffer(request.response);
    reader.onload = function (e) {
        gParam = new Float32Array(reader.result);
        document.getElementById("debug").innerHTML = 'Running network...<br /><br />';
        runNetwork('asm.js');
        runNetwork('weblas');
    }
};
request.send();

function runNetwork(backend) {
    const N = 19;
    const NN = N * N;
    const nFeaturePlane = 8;
    const nFilter = 128;

    const x = new BlinkArray();
    x.Init(backend);
    // weiqi (baduk, go) policy network in AlphaGo style
    // there are 8 one-hot encoded feature planes
    // 0: last_move, 1: four-and-above-liberty, 2: three-liberty, 3: two-liberty
    // 4: one-liberty, 5: empty, 6: opponent, 7: player
    x.nChannel = nFeaturePlane;
    x.data = new Float32Array(nFeaturePlane * NN);
    for (var i = 0; i < NN; i++)
        x.data[5 * NN + i] = 1; // empty board

    // timing begin...
    const t_start = new Date().getTime();

    // pre-act residual network with 6 residual blocks
    const bak = new Float32Array(nFilter * NN);
    x.Convolution(nFilter, 3)
    x.CopyTo(bak);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.Add(bak).CopyTo(bak);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.Add(bak).CopyTo(bak);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.Add(bak).CopyTo(bak);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.Add(bak).CopyTo(bak);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.Add(bak).CopyTo(bak);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.BatchNorm().ReLU().Convolution(nFilter, 3);
    x.Add(bak);
    x.BatchNorm().ReLU().Convolution(1, 1).Softmax();

    // timing end...
    const t_end = new Date().getTime();

    // debug output
    document.getElementById("debug").innerHTML += 'Backend: ' + backend + ', t = ' + ((t_end - t_start) * 0.001).toFixed(2) + 's';
    for (var i = 0; i < N; i++) {
        document.getElementById("debug").innerHTML += '<br />';
        for (var j = 0; j < N; j++) {
            document.getElementById("debug").innerHTML += ("      " + Math.round(100000.0 * x.data[i * N + j])).slice(-6);
        }
    }
    document.getElementById("debug").innerHTML += '<br /><br />';
}
