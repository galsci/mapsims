export LIBSHARP=~/miniconda
git clone https://github.com/Libsharp/libsharp --branch v1.0.0 --single-branch --depth 1 \
    && cd libsharp \
    && autoreconf \
    && CC="mpicc" \
    ./configure --enable-pic \
    && make -j4 \
    && cp -a auto/* $LIBSHARP \
    && cd python \
    && CC="gcc -g" LDSHARED="gcc -g -shared" \
    pip install .
