# BlinkDL

A minimalist deep learning library in Javascript using WebGL + asm.js. Runs in your browser. 

Currently it is a proof-of-concept (inference only).

The WebGL backend is powered by weblas: https://github.com/waylonflinn/weblas.

## Example

A weiqi (baduk, go) policy network in AlphaGo style:

<img width="500" alt="performance_image" src="https://user-images.githubusercontent.com/33809201/33012320-7659cb1e-ce1b-11e7-8c10-f63c56c1279d.png">

    const N = 19;
    const NN = N * N;
    const nFeaturePlane = 8;
    const nFilter = 128;

    const x = new BlinkArray();
    x.Init('weblas');
    x.nChannel = nFeaturePlane;
    x.data = new Float32Array(nFeaturePlane * NN);
    for (var i = 0; i < NN; i++)
        x.data[5 * NN + i] = 1; // set feature plane for empty board
    
    // pre-act residual network with 6 residual blocks
    const bak = new Float32Array(nFilter * NN);
    x.Convolution(nFilter, 3);
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
    
## Usage

    <script src='weblas.js' type='text/javascript'></script>
    <script src='BlinkDL.js' type='text/javascript'></script>
 
## Todo
- [x] Convolution (3x3_pad_1 and 1x1), BatchNorm, ReLU, Softmax
- [ ] Webworker and async
- [ ] Faster inference with weblas pipeline, WebGPU, WebAssembly
- [ ] Pooling layer
- [ ] FC layer
- [ ] Strided convolution
- [ ] Transposed convolution
- [ ] Training
