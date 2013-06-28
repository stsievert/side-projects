#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// the nice stuff python has
void getCol(float * in, int col, float * out, int width, int height);
void getRow(float * in, int row, float * out, int width, int height);
void putColIn(float * col, float colNumber, float * arr, int width, int height);
void putRowIn(float * row, float rowNumber, float * arr, int width, int height);
void copyPartsOfArray(float * in, float * out, int maxX, int maxY, int width, int height);
void copyPartsOfArrayInverse(float * in, float * out, int maxX, int maxY, int width, int height);
void copyArray(float * in, float * out, int len);
void swap(int * one, int * two);
void shuffle(int * in, int len);
void copySelectedInstances(float * in, float * out, int * indicies, int len);
float sign(float x);

// the actual dwt/idwt
float * dwt2(float * x, float * y, int width, int height);
void * dwt(float * x, int N, float * y);
float * dwt2_full(float * x,  int width, int height);
float * idwt(float * x, int N, float * y);
float * idwt2(float * x, float * y, int width, int height);
float * idwt2_full(float * x, int width, int height);
void IST();


int main(){
    // making our array
    IST();
    return 0;
}
void IST(){
    // making our array
    int i;
    int width = 256;
    int height = 256;
    int n = width * height;
    float p = 0.5;
    int upper = (int)(p * n);

    float * x = (float *)malloc(sizeof(float) * n);
    float * y = (float *)malloc(sizeof(float) * upper);
    float * xold = (float *)malloc(sizeof(float) * n);
    float * xold1 = (float *)malloc(sizeof(float) * n);
    float * t1 = (float *)malloc(sizeof(float) * n);
    float * temp = (float *)malloc(sizeof(float) * n);
    float * temp2 = (float *)malloc(sizeof(float) * n);
    float * temp3 = (float *)malloc(sizeof(float) * n);
    float * temp4 = (float *)malloc(sizeof(float) * n);
    for (i=0; i<n; i++){
        x[i] = i;
        xold[i] = 0;
        xold1[i] = 0;
    }
    int * rp = (int *)malloc(sizeof(int) * n);

    float tn = 1;
    float tn1;
    int its = 100;
    int k;
    float cut = 6.0;

    shuffle(rp, n);
    for (i=0; i<upper; i++){y[i] = x[rp[i]]; }

    /*printf("Start\n"); fflush(stdout);*/
    for (k=0; k<its; k++){
        /*printf("its: %d\n", k);*/
        tn1 = (1+sqrt(1+4*tn*tn))/2;

        for (i=0; i<n; i++){
            xold[i] = xold[i] + (tn-1)/tn1 * (xold[i] - xold1[i]);
        }

        copyArray(xold, t1, n);
        idwt2_full(t1, width, height);
        copySelectedInstances(t1, temp, rp, upper);

        for (i=0; i<upper; i++) temp2[i] = y[i] - temp[i];
        for (i=0; i<n; i++)     temp3[i] = 0;
        for (i=0; i<upper; i++) temp3[rp[i]] = temp2[i];
        dwt2_full(temp3, width, height);

        for (i=0; i<n; i++) xold[i] = temp3[i] + xold[i];
        for (i=0; i<n; i++){
            if (abs(xold[i]) < cut) xold[i] = 0;
            else xold[i] = xold[i] - sign(xold[i]) * cut;
        }

        for (i=0; i<n; i++){ xold1[i] = xold[i]; }
        tn = tn1; 

    }
    idwt2_full(xold, width, height);


}
float sign(float x){
    if      (x<0) return -1;
    else if (x>0) return 1;
    return 0;
}
void copySelectedInstances(float * in, float * out, int * indicies, int len){
    int i;
    for (i=0; i<len; i++){
        out[i] = in[indicies[i]];
    }
}
void copyArray(float * in, float * out, int len){
    int i;
    for(i=0; i<len; i++){
        out[i] = in[i];
    }
}

// functions that are mathier, but still one call in python
void shuffle(int * in, int len){
    // making solid indicies first
    int i;
    for (i=0; i<len; i++){
        in[i] = i;
    }
    for (i=0; i<len; i++){
        int r = (int)(rand() % len);
        /*printf("r: %d, in[i]: %d, in[r]: %d\n", r, in[i], in[r]);*/
        swap(&in[i], &in[r]);
        /*printf("r: %d, in[i]: %d, in[r]: %d\n\n", r, in[i], in[r]);*/
    }
}
void swap(int * one, int * two){
    int temp;
    temp = one[0];
    one[0] = two[0];
    two[0] = temp;
}


// BELOW:  dwt ,  dwt2  , dwt2_full
//        idwt , idwt2 , idwt2_full
//  dwt2_full and idwt2_full overwrite the input
void * dwt(float * x, int N, float * y){
    // works
    // N must be a factor of two
    int l = N/2;
    int i;
    for (i=0; i<l; i++){
        y[i] = x[2*i] + x[2*i + 1];
        y[i + l] = x[2*i] - x[2*i + 1];
    }
    for (i=0; i<N; i++){
        y[i] = y[i] / sqrt(2);
    }
}
float * dwt2(float * x, float * y, int width, int height){
    // x is our input
    // y is our output
    int i, j;

    // our row
    float * k  = (float *)malloc(sizeof(float) * width);
    float * k1 = (float *)malloc(sizeof(float) * width);

    for (i=0; i<height; i++){
        getRow(x, i, k, width, height);
        dwt(k, width, k1);
        putRowIn(k1, i, y, width, height);
    }
    for (i=0; i<width; i++){
        getCol(y, i, k, width, height);
        dwt(k, height, k1);
        putColIn(k1, i, y, width, height);
    }
}
float * dwt2_full(float * x, int width, int height){
    // overwrites x
    int i, j, k;
    int xx, yy;
    int order = log2(width);
    float * wavelet = (float *)malloc(sizeof(float) * height * width);
    float * waveletF = (float *)malloc(sizeof(float) * width * height);
    for (k=0; k<order; k++){
        // copy the array over 
        copyPartsOfArray(x, wavelet, width>>k, height>>k, width, height);
        
        // perform the dwt
        dwt2(wavelet, waveletF, width>>k, height>>k);

        // copy the array back
        copyPartsOfArrayInverse(waveletF, x, width>>k, height>>k, width, height);
    }
}
float * idwt(float * x, int N, float * y){
    int i;
    int le = N/2;
    for (i=0; i<le; i++){
        // we have (1+2) (3+4) (1-2) (3-4)
        y[2*i]   = (x[i] + x[i+le])/sqrt(2);
        y[2*i+1] = (x[i] - x[i+le])/sqrt(2);
    }
}
float * idwt2(float * x, float * y, int width, int height){
    int i;
    float * wavelet = (float *)malloc(sizeof(float) * width);
    float * wavelet1 = (float *)malloc(sizeof(float) * width);
    for (i=0; i<width; i++){
        getRow(x, i, wavelet, width, height);
        idwt(wavelet, height, wavelet1);
        putRowIn(wavelet1, i, y, width, height);
    }
    for (i=0; i<height; i++){
        getCol(y, i, wavelet, width, height);
        idwt(wavelet, width, wavelet1);
        putColIn(wavelet1, i, y, width, height);

    }
}
float * idwt2_full(float * x, int width, int height){
    // overwrites x
    int k;
    int order = (int)log2(width);
    float * wavelet  = (float *)malloc(sizeof(float) * width * height);
    float * waveletF = (float *)malloc(sizeof(float) * width * height);
    for (k=order; k>0; k--){
        // copy parts of array
        copyPartsOfArray(x, wavelet, width>>k-1, height>>k-1, width, height);
        // idwt2
        idwt2(wavelet, waveletF, width>>k-1, height>>k-1);
        // copy parts of array back in
        copyPartsOfArrayInverse(waveletF, x, width>>k-1, height>>k-1, width, height);
    }
}


// BELOW: the nice that C should have is below
void copyPartsOfArray(float * in, float * out, int maxX, int maxY, int width, int height){
    int x, y;
    int i=0;
    for (y=0; y<maxY; y++){
        for (x=0; x<maxX; x++){
            out[i] = in[y*width + x];
            i++;
        }
    }
}
void copyPartsOfArrayInverse(float * in, float * out, int maxX, int maxY, int width, int height){
    int x, y;
    int i=0;
    for (y=0; y<maxY; y++){
        for (x=0; x<maxX; x++){
            out[y*width + x] = in[i];
            i++;
        }
    }
}

void putRowIn(float * row, float rowNumber, float * arr, int width, int height){
    int i=0;
    int x, y;
    y = rowNumber;
    for (x=0; x<width; x++){
        arr[y*width+x] = row[x];
    }
}
void putColIn(float * col, float colNumber, float * arr, int width, int height){
    int i=0;
    int x, y;
    x = colNumber;
    for (y=0; y<height; y++){
        arr[y*width+x] = col[y];
    }
}
void getRow(float * in, int row, float * out, int width, int height){
    int i=0;
    int x, y;
    y = row;

    for(x=0; x<width; x++){
        out[x] = in[y*width + x];
    }
}
void getCol(float * in, int col, float * out, int width, int height){
    int i=0;
    int x, y;
    x = col;

    for(y=0; y<height; y++){
        out[y] = in[y*width + x];
    }
}
