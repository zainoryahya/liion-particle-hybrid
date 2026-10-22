// per-step cost of exact ZOH updates: modal (diagonal) form for n states; dense form; tridiagonal backward Euler (FV)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(){
  int ns[]={6,8,11,21,26,39,50,200,1000}; int nn=9; long steps=2000000; volatile double sink=0;
  printf("n,modal_ns,dense_ns,tridiagBE_ns\n");
  for(int t=0;t<nn;t++){int n=ns[t];
    double *a=malloc(n*sizeof(double)),*b=malloc(n*sizeof(double)),*c=malloc(n*sizeof(double)),*x=calloc(n,sizeof(double)),*y=calloc(n,sizeof(double));
    double *A=malloc((size_t)n*n*sizeof(double));
    for(int i=0;i<n;i++){a[i]=0.9+0.09*i/n;b[i]=1e-3;c[i]=1.0;} for(long i=0;i<(long)n*n;i++)A[i]=1e-3/n;
    long st= n>200? steps/50 : steps;
    double t0=now(); double q=1;
    for(long k=0;k<st;k++){double s=0; for(int i=0;i<n;i++){x[i]=a[i]*x[i]+b[i]*q; s+=c[i]*x[i];} sink+=s; q=-q;}
    double tm=(now()-t0)/st*1e9;
    long sd= n>50? st/ (n/10) : st; if(sd<1000) sd=1000;
    t0=now();
    for(long k=0;k<sd;k++){double s=0; for(int i=0;i<n;i++){double r=b[i]*q; for(int j=0;j<n;j++) r+=A[(long)i*n+j]*x[j]; y[i]=r;} for(int i=0;i<n;i++){x[i]=y[i]; s+=c[i]*x[i];} sink+=s; q=-q;}
    double td=(now()-t0)/sd*1e9;
    // Thomas algorithm, constant tridiagonal (prefactored) : forward/back substitution
    double *cp=malloc(n*sizeof(double)),*m=malloc(n*sizeof(double));
    for(int i=0;i<n;i++){cp[i]=-0.3; m[i]=1.0/1.7;}
    t0=now();
    for(long k=0;k<st;k++){ x[n-1]+=1e-3*q; y[0]=x[0]*m[0]; for(int i=1;i<n;i++) y[i]=(x[i]+0.3*y[i-1])*m[i]; x[n-1]=y[n-1]; for(int i=n-2;i>=0;i--) x[i]=y[i]-cp[i]*x[i+1]; sink+=x[n-1]; q=-q;}
    double tt=(now()-t0)/st*1e9;
    printf("%d,%.1f,%.1f,%.1f\n",n,tm,td,tt);
    free(a);free(b);free(c);free(x);free(y);free(A);free(cp);free(m);}
  fprintf(stderr,"%g\n",sink); return 0;}
