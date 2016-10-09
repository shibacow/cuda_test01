# CUDAtest


aws g2.2xlarge cuda test

gpu cudaのテストを行うcudaのインストール先は,/opt

こちらのリンクに乗っているcudaのサンプルプログラムを動かす

http://www.gdep.jp/page/view/251

3番目のQ3 matrixは動かなかったので、

http://hidemon-memo.blogspot.jp/2014/10/cuda.html

こちらをサンプルにして作成した。

helper_cuda.hが無かったので、こちらを使い、pathを通した。

https://github.com/pathscale/nvidia_sdk_samples


deviceQuery/common/inc
にpathを通す(Makefile参照)。


```cuda
  cudaevent_t start;
```

を


```cuda
  cudaEvent_t start;
```

に修正した。

make allでmatrix_gpu.exeとmatrix_cpu.exeが出来る。それぞれgpuとcpuを使っている

MakefileのINCLUDE= -I/foo/bar/common/incはそちらのもの

