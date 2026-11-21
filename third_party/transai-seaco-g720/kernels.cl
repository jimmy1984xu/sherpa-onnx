
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
inline float sgadd(float x,__local float*t){
 float a=sub_group_reduce_add(x);if(get_sub_group_local_id()==0)t[get_sub_group_id()]=a;
 barrier(CLK_LOCAL_MEM_FENCE);float y=0;for(uint i=0;i<get_num_sub_groups();i++)y+=t[i];barrier(CLK_LOCAL_MEM_FENCE);return y;
}
inline float sgmax(float x,__local float*t){
 float a=sub_group_reduce_max(x);if(get_sub_group_local_id()==0)t[get_sub_group_id()]=a;
 barrier(CLK_LOCAL_MEM_FENCE);float y=-INFINITY;for(uint i=0;i<get_num_sub_groups();i++)y=fmax(y,t[i]);barrier(CLK_LOCAL_MEM_FENCE);return y;
}
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_arm_matrix_multiply : enable
typedef half ACT;
#define ENC_SCALE 0.0078125f
// Cooperative MMUL lane mapping follows the MIT Arm Compute Library pattern
// already used by sherpa-g720/native/generated/teacher_kernels.cl.
int nr(__global const int *dims,int kind,int cap){return kind<0?cap:dims[kind];}
__kernel void norm(__global const ACT*x,__global const float*w,ulong go,ulong bo,
 __global ACT*y,int R,int C,float eps,__global const int*d,int kind){__local float sgtemp[64];
 int r=get_group_id(0),l=get_local_id(0);if(r>=nr(d,kind,R))return;float s=0;
 for(int c=l;c<C;c+=64)s+=x[r*C+c];float mean=sgadd(s,sgtemp)/C;
 s=0;for(int c=l;c<C;c+=64){float z=x[r*C+c]-mean;s+=z*z;}float inv=rsqrt(sgadd(s,sgtemp)/C+eps);
 for(int c=l;c<C;c+=64)y[r*C+c]=(x[r*C+c]-mean)*inv*w[go+c]+w[bo+c];
}
__kernel void quantvec(__global const ACT*x,__global char*q,__global float*s,int R,int C,int K,
 __global const int*d,int kind){__local float sgtemp[64];
 int r=get_group_id(0),l=get_local_id(0);if(r>=nr(d,kind,R))return;float m=0;
 for(int j=l;j<C/4;j+=64){float4 a=fabs(convert_float4(vload4(0,x+r*C+j*4)));m=fmax(m,fmax(fmax(a.x,a.y),fmax(a.z,a.w)));}
 float sc=fmax(sgmax(m,sgtemp)/127.f,1e-12f);if(!l)s[r]=sc;
 for(int j=l;j<K/4;j+=64){char4 a=j<C/4?convert_char4_sat_rte(convert_float4(vload4(0,x+r*C+j*4))/sc):(char4)(0);vstore4(a,0,q+r*K+j*4);}
}
// Four adjacent columns share one activation tile and one packed weight load.
inline int4 mmul_columns(char4 lhs,char16 rhs,int4 acc) {
 return (int4)(arm_matrix_multiply(lhs,rhs.s0123,acc.x),
               arm_matrix_multiply(lhs,rhs.s4567,acc.y),
               arm_matrix_multiply(lhs,rhs.s89ab,acc.z),
               arm_matrix_multiply(lhs,rhs.scdef,acc.w));
}
inline void store_columns(int4 acc,__global ACT *y,__global const float *as,
 __global const float *weights,ulong so,ulong bo,int r,int c,int N,int relu,float outscale) {
 float4 z=convert_float4(acc)*as[r]*vload4(0,weights+so+c)+vload4(0,weights+bo+c);
 if(relu) z=fmax(z,0.f);
 vstore4(convert_half4_rte(z*outscale),0,y+r*N+c);
}
inline void gemm8_tiled(__global const char *a,__global const char *w,ulong wo,__global const float *as,
 __global const float *fw,ulong so,ulong bo,__global ACT *y,int R,int K,int N,int relu,
 __global const int *d,int kind,float outscale,int stage_k,__local char16 *stage) {
 int lane=get_local_id(0),subrow=get_local_id(1),subrows=get_local_size(1);
 int first=get_group_id(1)*subrows*4,sy=first+subrow*4,r=sy+lane/4;
 int rows=nr(d,kind,R),c=get_group_id(0)*16+(lane&3)*4;
 if(first>=rows)return;
 const __global char *weights=w+wo+get_group_id(0)*K*16;
 int4 acc=(int4)(0);
 for(int base=0;base<K;base+=stage_k) {
   int count=min(stage_k,K-base);
   for(int z=subrow*16+lane;z<count;z+=subrows*16) stage[z]=vload16(z,weights+base*16);
   barrier(CLK_LOCAL_MEM_FENCE);
   if(sy<rows) {
     for(int j=0;j<count;j+=16) {
       char4 lhs=r<rows?vload4(lane&3,a+r*K+base+j):(char4)(0);
       acc=mmul_columns(lhs,stage[j+lane],acc);
     }
   }
   barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(r<rows&&c<N)store_columns(acc,y,as,fw,so,bo,r,c,N,relu,outscale);
}
__kernel void gemm8(__global const char *a,__global const char *w,ulong wo,__global const float *as,
 __global const float *fw,ulong so,ulong bo,__global ACT *y,int R,int K,int N,int NP,int relu,
 __global const int *d,int kind,float outscale) {
 __local char16 stage[128];
 gemm8_tiled(a,w,wo,as,fw,so,bo,y,R,K,N,relu,d,kind,outscale,128,stage);
}
__kernel void gemm8_large(__global const char *a,__global const char *w,ulong wo,__global const float *as,
 __global const float *fw,ulong so,ulong bo,__global ACT *y,int R,int K,int N,int NP,int relu,
 __global const int *d,int kind,float outscale) {
 __local char16 stage[512];
 gemm8_tiled(a,w,wo,as,fw,so,bo,y,R,K,N,relu,d,kind,outscale,512,stage);
}
__kernel __attribute__((reqd_work_group_size(16,1,1)))
void gemmh(__global const ACT*a,__global const half*w,ulong wo,__global const float*fw,ulong bo,
 __global ACT*y,int R,int K,int KP,int N,int relu,__global const int*d,int kind,float outscale){
 int l=get_local_id(0),tx=l&3,ty=l>>2,r=get_group_id(1)*16+ty*4,c=get_group_id(0)*16+tx*4;
 int rows=nr(d,kind,R);if(get_group_id(1)*16>=rows)return;float acc[4][4]={{0}};
 for(int k=0;k<KP;k+=4){half lhs[4],rhs[4];
  #pragma unroll
  for(int i=0;i<4;i++){
  lhs[i]=r+i<rows&&k+tx<K?convert_half_rte(a[(r+i)*K+k+tx]):(half)0;
  rhs[i]=c+i<N?w[wo+((c+i)/16)*KP*16+(k/4)*64+ty*16+(c+i)%16]:(half)0;}
  #pragma unroll
  for(int i=0;i<4;i++){
   #pragma unroll
   for(int j=0;j<4;j++)acc[i][j]=arm_matrix_multiply(lhs[i],rhs[j],acc[i][j]);}
 }
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)if(r+i<rows&&c+j<N){float z=acc[i][j]+fw[bo+c+j];y[(r+i)*N+c+j]=(relu?fmax(z,0.f):z)*outscale;}
}
__kernel void gemm8direct(__global const char *a,__global const char *w,ulong wo,__global const float *as,
 __global const float *fw,ulong so,ulong bo,__global ACT *y,int R,int K,int N,int NP,int relu,
 __global const int *d,int kind,float outscale) {
 int lane=get_local_id(0),sy=(get_group_id(1)*get_local_size(1)+get_local_id(1))*4;
 int r=sy+lane/4,c=get_group_id(0)*16+(lane&3)*4,rows=nr(d,kind,R);
 if(sy>=rows)return;
 const __global char *weights=w+wo+get_group_id(0)*K*16+lane*16;
 int4 acc=(int4)(0);
 for(int k=0;k<K;k+=16) {
   char4 lhs=r<rows?vload4(lane&3,a+r*K+k):(char4)(0);
   acc=mmul_columns(lhs,vload16(k,weights),acc);
 }
 if(r<rows&&c<N)store_columns(acc,y,as,fw,so,bo,r,c,N,relu,outscale);
}
__kernel void add(__global const ACT*a,__global const ACT*b,__global ACT*y,int R,int C,
 __global const int*d,int kind){int i=get_global_id(0);if(i<nr(d,kind,R)*C)y[i]=a[i]+b[i];}
__kernel void encoder_residual_qnorm(__global const ACT *a,__global const ACT *b,
 __global const ACT *residual,__global ACT *out,__global char *q,__global float *scales,
 __global const float *weights,ulong gain,ulong bias,__global const int *dims,int operation) {
 int row=get_group_id(0),lane=get_local_id(0);
 if(row>=dims[0]) return;
 __local float scratch[64];
 float values[8],sum=0.f;
 for(int c=lane;c<512;c+=64) {
   int index=row*512+c;
   float add=(float)a[index]+(float)b[index];
   ACT rounded=operation==0?add:add*ENC_SCALE+(operation==2?(float)residual[index]:0.f);
   out[index]=rounded;
   values[c/64]=(float)rounded;
   sum+=values[c/64];
 }
 float mean=sgadd(sum,scratch)/512;
 sum=0.f;
 for(int c=lane;c<512;c+=64) {float z=values[c/64]-mean;sum+=z*z;}
 float inv=rsqrt(sgadd(sum,scratch)/512+1.e-12f),maximum=0.f;
 for(int c=lane;c<512;c+=64) {
   float z=(values[c/64]-mean)*inv*weights[gain+c]+weights[bias+c];
   values[c/64]=z;
   maximum=fmax(maximum,fabs(z));
 }
 float scale=fmax(sgmax(maximum,scratch)/127.f,1.e-12f);
 if(lane==0) scales[row]=scale;
 for(int c=lane;c<512;c+=64) q[row*512+c]=convert_char_sat_rte(values[c/64]/scale);
}
// SANM weights are stored tap-major for a coalesced four-channel read.
__kernel void depth(__global const ACT *x,__global const float *w,ulong wo,__global ACT *y,
 int R,int C,int stride,int offset,int taps,int pad,__global const int *d,int kind) {
 int col=get_global_id(0),r=get_global_id(1),rows=nr(d,kind,R);
 if(r>=rows||col>=C/4)return;
 float4 value=convert_float4(vload4(col,x+r*stride+offset));
 for(int k=0;k<taps;k++) {
   int t=r+k-pad;
   if(t>=0&&t<rows&&(kind!=5||((r<d[1])==(t<d[1])))) {
     float4 samples=convert_float4(vload4(col,x+t*stride+offset));
     value+=samples*vload4(col,w+wo+k*C);
   }
 }
 vstore4(convert_half4_rte(value),col,y+r*C);
}
__kernel void quant_head(__global const ACT*x,__global char*q,__global float*s,
 int R,int stride,int offset,__global const int*d,int kind){
 int r=get_group_id(0)%R,h=get_group_id(0)/R,l=get_local_id(0);if(r>=nr(d,kind,R))return;
 __local float v[64];float a=x[r*stride+offset+h*128+l],b=x[r*stride+offset+h*128+l+64];
 float sc=fmax(sgmax(fmax(fabs(a),fabs(b)),v)/127.f,1e-12f);
 int row=h*R+r;if(!l)s[row]=sc;q[row*128+l]=convert_char_sat_rte(a/sc);q[row*128+l+64]=convert_char_sat_rte(b/sc);
}
__kernel __attribute__((reqd_work_group_size(16,1,1)))
void score8(__global const char*q,__global const char*k,__global const float*qs,__global const float*ks,
 __global float*s,int R,int M,__global const int*d,int rk,int mk){
 int l=get_local_id(0),tx=l&3,ty=l>>2,r=get_group_id(1)*16+ty*4,c=get_group_id(0)*16+tx*4,h=get_group_id(2);
 int rows=nr(d,rk,R),mem=nr(d,mk,M);if(get_group_id(1)*16>=rows||get_group_id(0)*16>=mem)return;
 int acc[4][4]={{0}};
 for(int j=0;j<128;j+=16){char4 lhs[4],rhs[4];
 #pragma unroll
 for(int i=0;i<4;i++){lhs[i]=r+i<rows?vload4(0,q+(h*R+r+i)*128+j+tx*4):(char4)(0);rhs[i]=c+i<mem?vload4(0,k+(h*M+c+i)*128+j+ty*4):(char4)(0);}
 #pragma unroll
 for(int i=0;i<4;i++){
 #pragma unroll
 for(int n=0;n<4;n++)acc[i][n]=arm_matrix_multiply(lhs[i],rhs[n],acc[i][n]);}}
 for(int i=0;i<4;i++)for(int n=0;n<4;n++)if(r+i<rows&&c+n<mem)s[(h*R+r+i)*M+c+n]=acc[i][n]*qs[h*R+r+i]*ks[h*M+c+n]*0.08838834764831845f;
}
__kernel __attribute__((reqd_work_group_size(16,1,1)))
void score(__global const ACT*q,__global const ACT*kv,__global float*s,int R,int M,
 int qs,int qo,int ks,int ko,__global const int*d,int rk,int mk){
 int l=get_local_id(0),tx=l&3,ty=l>>2,r=get_group_id(1)*16+ty*4,c=get_group_id(0)*16+tx*4,h=get_group_id(2);
 int rows=nr(d,rk,R),mem=nr(d,mk,M);if(get_group_id(1)*16>=rows||get_group_id(0)*16>=mem)return;
 float acc[4][4]={{0}};for(int k=0;k<128;k+=4){half lhs[4],rhs[4];
 #pragma unroll
 for(int i=0;i<4;i++){
 lhs[i]=r+i<rows?convert_half_rte(q[(r+i)*qs+qo+h*128+k+tx]):(half)0;
 rhs[i]=c+i<mem?convert_half_rte(kv[(c+i)*ks+ko+h*128+k+ty]):(half)0;}
 #pragma unroll
 for(int i=0;i<4;i++){
 #pragma unroll
 for(int j=0;j<4;j++)acc[i][j]=arm_matrix_multiply(lhs[i],rhs[j],acc[i][j]);}}
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)if(r+i<rows&&c+j<mem)s[(h*R+r+i)*M+c+j]=acc[i][j]*0.08838834764831845f;
}
__kernel void softmax(__global float*s,int R,int M,__global const int*d,int rk,int mk){
 int l=get_local_id(0),r=get_group_id(0),h=get_group_id(1);if(r>=nr(d,rk,R))return;
 int mem=nr(d,mk,M),off=(h*R+r)*M;__local float v[64];float a=-INFINITY;
 for(int j=l;j<mem;j+=64)a=fmax(a,s[off+j]);v[l]=a;barrier(CLK_LOCAL_MEM_FENCE);
 for(int j=32;j;j>>=1){if(l<j)v[l]=fmax(v[l],v[l+j]);barrier(CLK_LOCAL_MEM_FENCE);}a=v[0];
 barrier(CLK_LOCAL_MEM_FENCE);float b=0;
 for(int j=l;j<mem;j+=64)b+=exp(s[off+j]-a);v[l]=b;barrier(CLK_LOCAL_MEM_FENCE);
 for(int j=32;j;j>>=1){if(l<j)v[l]+=v[l+j];barrier(CLK_LOCAL_MEM_FENCE);}b=v[0];
 for(int j=l;j<mem;j+=64)s[off+j]=exp(s[off+j]-a)/b;
}
__kernel void quant_v(__global const ACT*x,__global char*q,__global float*s,int M,int MP,int stride,int offset,__global const int*d,int kind){
 int r=get_group_id(0)%M,h=get_group_id(0)/M,l=get_local_id(0);if(r>=nr(d,kind,M))return;
 float a=x[r*stride+offset+h*128+l],b=x[r*stride+offset+h*128+l+64];__local float v[64];float sc=fmax(sgmax(fmax(fabs(a),fabs(b)),v)/127.f,1e-12f);if(!l)s[h*M+r]=sc;
 int ka=((((h*32+l/4)*(MP/16)+r/16)*4+l%4)*16)+r%16;
 int kb=((((h*32+(l+64)/4)*(MP/16)+r/16)*4+(l+64)%4)*16)+r%16;
 q[ka]=convert_char_sat_rte(a/sc);q[kb]=convert_char_sat_rte(b/sc);
}
// Quantize all three encoder projections in one dispatch. Each group
// uses the same per-head maximum and scale as quant_head/quant_v.
__kernel void quant_encoder_qkv(__global const ACT *x,
 __global char *q, __global float *qs, __global char *k, __global float *ks,
 __global char *v, __global float *vs, int R, int MP, __global const int *d) {
 int group = get_group_id(0), lane = get_local_id(0);
 int row = group % R, head = (group / R) % 4, projection = group / (4 * R);
 if (row >= nr(d, 0, R)) return;
 int index = row * 1536 + projection * 512 + head * 128 + lane;
 float a = x[index], b = x[index + 64];
 __local float maxima[64];
 float scale = fmax(sgmax(fmax(fabs(a), fabs(b)), maxima) / 127.f, 1.e-12f);
 char qa = convert_char_sat_rte(a / scale), qb = convert_char_sat_rte(b / scale);
 int head_row = head * R + row;
 if (projection < 2) {
   __global char *output = projection == 0 ? q : k;
   __global float *scales = projection == 0 ? qs : ks;
   if (lane == 0) scales[head_row] = scale;
   output[head_row * 128 + lane] = qa;
   output[head_row * 128 + lane + 64] = qb;
 } else {
   if (lane == 0) vs[head_row] = scale;
   int offset = ((((head * 32 + lane / 4) * (MP / 16) + row / 16) * 4 + lane % 4) * 16) + row % 16;
   v[offset] = qa;
   v[offset + MP * 64] = qb;
 }
}
__kernel void prob8(__global const float*s,__global const float*vs,__global char*q,__global float*qs,
 int R,int M,int MP,__global const int*d,int rk,int mk){__local float sgtemp[64];
 int r=get_group_id(0)%R,h=get_group_id(0)/R,l=get_local_id(0);if(r>=nr(d,rk,R))return;int mem=nr(d,mk,M),off=(h*R+r)*M;
 float m=-INFINITY;for(int j=l;j<mem;j+=64)m=fmax(m,s[off+j]);m=sgmax(m,sgtemp);
 float sum=0,mw=0;for(int j=l;j<mem;j+=64){float e=exp(s[off+j]-m);sum+=e;mw=fmax(mw,e*vs[h*M+j]);}
 sum=sgadd(sum,sgtemp);mw=sgmax(mw,sgtemp);
 float sc=fmax(mw/127.f,1e-12f);if(!l)qs[h*R+r]=sc/sum;
 for(int j=l;j<MP;j+=64)q[(h*R+r)*MP+j]=j<mem?convert_char_sat_rte(exp(s[off+j]-m)*vs[h*M+j]/sc):0;
}
__kernel __attribute__((reqd_work_group_size(16,1,1)))
void context8(__global const char*a,__global const char*v,__global const float*ps,__global ACT*y,
 int R,int M,int MP,__global const int*d,int rk,int mk){
 int l=get_local_id(0),tx=l&3,ty=l>>2,r=get_group_id(1)*16+ty*4,c=get_group_id(0)*16+tx*4,h=get_group_id(2),rows=nr(d,rk,R);
 if(get_group_id(1)*16>=rows)return;int acc[4][4]={{0}};
 for(int k=0;k<MP;k+=16){char4 lhs[4],rhs[4];
 #pragma unroll
 for(int i=0;i<4;i++){lhs[i]=r+i<rows?vload4(0,a+(h*R+r+i)*MP+k+tx*4):(char4)(0);
  rhs[i]=vload4(0,v+((((h*32+(c+i)/4)*(MP/16)+k/16)*4+(c+i)%4)*16)+ty*4);}
 #pragma unroll
 for(int i=0;i<4;i++){
 #pragma unroll
 for(int j=0;j<4;j++)acc[i][j]=arm_matrix_multiply(lhs[i],rhs[j],acc[i][j]);}}
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)if(r+i<rows)y[(r+i)*512+h*128+c+j]=acc[i][j]*ps[h*R+r+i];
}
__kernel void position(__global const ACT*x,__global ACT*y,int R){int c=get_global_id(0),r=get_global_id(1);if(c>=560||r>=R)return;
 float a=(r+1)*exp((c%280)*(-log(10000.f)/279.f));y[r*560+c]=((float)x[r*560+c]*sqrt(512.f)+(c<280?sin(a):cos(a)))*ENC_SCALE;}
__kernel void unfold(__global const ACT*x,__global ACT*y,int R,__global const int*d){R=d[0];int c=get_global_id(0),r=get_global_id(1);if(c>=1536||r>=R)return;int t=r+c%3-1;y[r*1536+c]=t>=0&&t<R?x[t*512+c/3]:0;}
__kernel void alpha_project(__global const ACT*x,__global const float*wf,ulong wo,ulong bo,__global float*a,int R){
 int r=get_group_id(0),l=get_local_id(0);__global const half*w=(__global const half*)wf;__local float v[64];float s=0;
 for(int k=l;k<512;k+=64)s+=(float)x[r*512+k]*(float)w[wo+(k/4)*64+(k%4)*16];v[l]=s;barrier(CLK_LOCAL_MEM_FENCE);
 for(int j=32;j;j>>=1){if(l<j)v[l]+=v[l+j];barrier(CLK_LOCAL_MEM_FENCE);}if(!l)a[r]=1.f/(1.f+exp(-(v[0]+wf[bo])));
}
__kernel void cif(__global const ACT*x,__global const float*a,__global ACT*y,__global int*d,int R,float tail){
 int c=get_global_id(0);if(c>=512)return;float integrate=0,frame=0;int n=0;
 for(int t=0;t<d[0];t++){float al=a[t],completion=1.f-integrate;integrate+=al;int fire=integrate>=1.f;float cur=fire?completion:al;frame+=cur*x[t*512+c];if(fire){y[n*512+c]=frame;n++;integrate-=1.f;frame=(al-cur)*x[t*512+c];}}
 integrate+=tail;if(integrate>=1.f){y[n*512+c]=frame;n++;}if(c==0){d[1]=n;d[5]=2*n;}
}
__kernel void choose(__global const ACT*base,__global const ACT*hw,__global int*res,
 int R,int N,int use_hw,int nobias,__global const int*d,int score_offset){int r=get_group_id(0),l=get_local_id(0);if(r==0&&l==0)res[0]=d[1];if(r>=d[1])return;
 __local float bv[64],hv[64];__local int bi[64],hi[64];float bmax=-INFINITY,hmax=-INFINITY;int bidx=0,hidx=0;
 for(int c=l;c<N;c+=64){float b=base[r*N+c];if(b>bmax){bmax=b;bidx=c;}if(use_hw){float h=hw[r*N+c];if(h>hmax){hmax=h;hidx=c;}}}
 bv[l]=bmax;hv[l]=hmax;bi[l]=bidx;hi[l]=hidx;barrier(CLK_LOCAL_MEM_FENCE);
 for(int j=32;j;j>>=1){if(l<j){if(bv[l+j]>bv[l]||(bv[l+j]==bv[l]&&bi[l+j]<bi[l])){bv[l]=bv[l+j];bi[l]=bi[l+j];}if(hv[l+j]>hv[l]||(hv[l+j]==hv[l]&&hi[l+j]<hi[l])){hv[l]=hv[l+j];hi[l]=hi[l+j];}}barrier(CLK_LOCAL_MEM_FENCE);}
 // Preserve NO_BIAS/argmax exactly; return the selected distribution's
 // normalized log probability, without a separate context reward.
 int contextual=use_hw&&hi[0]!=nobias;
 float maximum=contextual?hv[0]:bv[0];
 barrier(CLK_LOCAL_MEM_FENCE);
 float sum=0;for(int c=l;c<N;c+=64)sum+=exp((contextual?hw[r*N+c]:base[r*N+c])-maximum);
 bv[l]=sum;barrier(CLK_LOCAL_MEM_FENCE);
 for(int j=32;j;j>>=1){if(l<j)bv[l]+=bv[l+j];barrier(CLK_LOCAL_MEM_FENCE);}
 if(l==0){res[r+1]=contextual?hi[0]:bi[0];((__global float*)(res+score_offset))[r]=-log(bv[0]);}
}
// GPU PCM -> Kaldi-compatible fbank, followed by LFR7/6 and global CMVN.
__kernel void fft(__global const short*pcm,__global const float*window,__global float*power,int samples,int frames,__global const int*d){
 int t=get_group_id(0),l=get_local_id(0);if(t>=d[4])return;__local float2 a[512];__local float red[256];
 float s=0;for(int j=l;j<400;j+=256)s+=(float)pcm[t*160+j];red[l]=s;barrier(CLK_LOCAL_MEM_FENCE);
 for(int j=128;j;j>>=1){if(l<j)red[l]+=red[l+j];barrier(CLK_LOCAL_MEM_FENCE);}float mean=red[0]/400.f;
 for(int j=l;j<512;j+=256){float z=0;if(j<400){float p=(float)pcm[t*160+max(j-1,0)]-mean;z=((float)pcm[t*160+j]-mean-0.97f*p)*window[j];}
  uint v=j,b=0;for(int k=0;k<9;k++){b=(b<<1)|(v&1);v>>=1;}a[b]=(float2)(z,0);}
 barrier(CLK_LOCAL_MEM_FENCE);
 for(int len=2;len<=512;len<<=1){int halfn=len>>1,base=(l/halfn)*len,j=l%halfn;float ang=-6.283185307179586f*j/len;
  float2 u=a[base+j],v=a[base+j+halfn];float co=cos(ang),si=sin(ang);float2 z=(float2)(v.x*co-v.y*si,v.x*si+v.y*co);
  barrier(CLK_LOCAL_MEM_FENCE);a[base+j]=u+z;a[base+j+halfn]=u-z;barrier(CLK_LOCAL_MEM_FENCE);}
 power[t*257+l]=dot(a[l],a[l]);if(l==0)power[t*257+256]=dot(a[256],a[256]);
}
__kernel void mel(__global const float*p,__global const float*weights,ulong wo,__global float*y,int frames,__global const int*ranges,__global const int*d){frames=d[4];
 int m=get_global_id(0),t=get_global_id(1);if(m>=80||t>=frames)return;float s=0;for(int k=ranges[m*2];k<ranges[m*2+1];k++)s+=p[t*257+k]*weights[wo+m*257+k];y[t*80+m]=log(fmax(s,1.1920928955078125e-7f));}
__kernel void lfr(__global const float*x,__global const float*w,ulong co,__global ACT*y,int frames,int R,__global const int*d){frames=d[4];R=d[0];int c=get_global_id(0),r=get_global_id(1);if(c>=560||r>=R)return;
 int t=clamp(r*6+c/80-3,0,frames-1);y[r*560+c]=(x[t*80+c%80]+w[co+c])*w[co+560+c];}
__kernel void lstmin(__global const float*w,ulong eo,__global const int*ids,__global const int*lens,
 __global const ACT*lower,__global const ACT*h,__global ACT*y,int B,int t,int layer){
 int c=get_global_id(0),b=get_global_id(1);if(c>=1024||b>=B)return;
 y[b*1024+c]=c>=512?h[b*512+c-512]:(layer?lower[b*512+c]:w[eo+ids[b*10+t]*512+c]);}
__kernel void lstmstep(__global const ACT*g,__global ACT*h,__global float*cell,
 __global const int*lens,int B,int t){int c=get_global_id(0),b=get_global_id(1);if(c>=512||b>=B||t>=lens[b])return;
 int o=b*2048+c;float i=1.f/(1.f+exp(-g[o])),f=1.f/(1.f+exp(-g[o+512]));float z=tanh(g[o+1024]),out=1.f/(1.f+exp(-g[o+1536]));
 float v=f*cell[b*512+c]+i*z;cell[b*512+c]=v;h[b*512+c]=out*tanh(v);}
__kernel void importance(__global const float*s,__global float*y,int R,int M,__global const int*d){
 int m=get_global_id(0);if(m>=M)return;float v=0;for(int h=0;h<4;h++)for(int r=0;r<d[1];r++)v+=s[(h*R+r)*M+m];y[m]=v;}
__kernel void select_hotwords(__global const float*score,__global int*indices,__global int*d,int M){
 if(get_global_id(0))return;for(int j=0;j<50;j++){int best=-1;float v=-INFINITY;for(int m=0;m<M;m++){
  int used=0;for(int k=0;k<j;k++)if(indices[k]==m)used=1;
  if(!used&&(score[m]>v||(score[m]==v&&(best<0||m<best)))){v=score[m];best=m;}}
  indices[j]=best;}
 indices[50]=M-1;d[2]=51;
}
__kernel void gather(__global const ACT*x,__global const int*indices,__global ACT*y,int C){
 int c=get_global_id(0),r=get_global_id(1);if(c<C&&r<51)y[r*C+c]=x[indices[r]*C+c];}
__kernel void text_result(__global int*res,__global const uchar*tokens,__global const int*offsets,int R){
 if(get_global_id(0))return;__global uchar*out=(__global uchar*)(res+R+2);int n=res[0],pos=0,merge=0,last_ascii=0;
 for(int i=0;i<n;i++){int id=res[i+1];if(id==0||id==1||id==2)continue;int a=offsets[id],len=offsets[id+1]-a;
  int ascii=tokens[a]<128,continuation=len>=2&&tokens[a+len-1]=='@'&&tokens[a+len-2]=='@';
  if(continuation){len-=2;if(!merge&&pos)out[pos++]=' ';}
  else if((ascii&&!merge)||(!ascii&&last_ascii)){if(pos)out[pos++]=' ';}
  for(int j=0;j<len;j++)out[pos++]=tokens[a+j];merge=continuation;last_ascii=ascii;
 }out[pos]=0;res[R+1]=pos;
}


// BEGIN DIMENSION SPECIALIZATIONS
__kernel void qnorm_512(__global const ACT*x,__global const float*w,ulong go,ulong bo,
 __global char*q,__global float*scale,int R,int suppliedC,int K,__global const int*d,int kind){__local float sgtemp[64];
 const int C=512;int r=get_group_id(0),l=get_local_id(0);if(r>=nr(d,kind,R))return;
 float values[8],s=0;
 for(int c=l;c<C;c+=64)s+=x[r*C+c];float mean=sgadd(s,sgtemp)/C;
 s=0;for(int c=l;c<C;c+=64){float z=x[r*C+c]-mean;s+=z*z;}float inv=rsqrt(sgadd(s,sgtemp)/C+1.e-12f);
 s=0;for(int c=l;c<C;c+=64){float z=(x[r*C+c]-mean)*inv*w[go+c]+w[bo+c];values[c/64]=z;s=fmax(s,fabs(z));}
 float sc=fmax(sgmax(s,sgtemp)/127.f,1e-12f);if(!l)scale[r]=sc;
 for(int c=l;c<K;c+=64)q[r*K+c]=c<C?convert_char_sat_rte(values[c/64]/sc):0;
}

__kernel void qnorm_560(__global const ACT*x,__global const float*w,ulong go,ulong bo,
 __global char*q,__global float*scale,int R,int suppliedC,int K,__global const int*d,int kind){__local float sgtemp[64];
 const int C=560;int r=get_group_id(0),l=get_local_id(0);if(r>=nr(d,kind,R))return;
 float values[9],s=0;
 for(int c=l;c<C;c+=64)s+=x[r*C+c];float mean=sgadd(s,sgtemp)/C;
 s=0;for(int c=l;c<C;c+=64){float z=x[r*C+c]-mean;s+=z*z;}float inv=rsqrt(sgadd(s,sgtemp)/C+1.e-12f);
 s=0;for(int c=l;c<C;c+=64){float z=(x[r*C+c]-mean)*inv*w[go+c]+w[bo+c];values[c/64]=z;s=fmax(s,fabs(z));}
 float sc=fmax(sgmax(s,sgtemp)/127.f,1e-12f);if(!l)scale[r]=sc;
 for(int c=l;c<K;c+=64)q[r*K+c]=c<C?convert_char_sat_rte(values[c/64]/sc):0;
}

__kernel void qnorm_1024(__global const ACT*x,__global const float*w,ulong go,ulong bo,
 __global char*q,__global float*scale,int R,int suppliedC,int K,__global const int*d,int kind){__local float sgtemp[64];
 const int C=1024;int r=get_group_id(0),l=get_local_id(0);if(r>=nr(d,kind,R))return;
 float values[16],s=0;
 for(int c=l;c<C;c+=64)s+=x[r*C+c];float mean=sgadd(s,sgtemp)/C;
 s=0;for(int c=l;c<C;c+=64){float z=x[r*C+c]-mean;s+=z*z;}float inv=rsqrt(sgadd(s,sgtemp)/C+1.e-12f);
 s=0;for(int c=l;c<C;c+=64){float z=(x[r*C+c]-mean)*inv*w[go+c]+w[bo+c];values[c/64]=z;s=fmax(s,fabs(z));}
 float sc=fmax(sgmax(s,sgtemp)/127.f,1e-12f);if(!l)scale[r]=sc;
 for(int c=l;c<K;c+=64)q[r*K+c]=c<C?convert_char_sat_rte(values[c/64]/sc):0;
}

__kernel void qnorm_2048(__global const ACT*x,__global const float*w,ulong go,ulong bo,
 __global char*q,__global float*scale,int R,int suppliedC,int K,__global const int*d,int kind){__local float sgtemp[64];
 const int C=2048;int r=get_group_id(0),l=get_local_id(0);if(r>=nr(d,kind,R))return;
 float values[32],s=0;
 for(int c=l;c<C;c+=64)s+=x[r*C+c];float mean=sgadd(s,sgtemp)/C;
 s=0;for(int c=l;c<C;c+=64){float z=x[r*C+c]-mean;s+=z*z;}float inv=rsqrt(sgadd(s,sgtemp)/C+1.e-12f);
 s=0;for(int c=l;c<C;c+=64){float z=(x[r*C+c]-mean)*inv*w[go+c]+w[bo+c];values[c/64]=z;s=fmax(s,fabs(z));}
 float sc=fmax(sgmax(s,sgtemp)/127.f,1e-12f);if(!l)scale[r]=sc;
 for(int c=l;c<K;c+=64)q[r*K+c]=c<C?convert_char_sat_rte(values[c/64]/sc):0;
}
// Pack two independent SeACo sequences densely, keeping both SANM boundaries.
__kernel void stack_branches(__global const ACT*a,__global const ACT*b,__global ACT*y,int R,__global const int*d){
 int i=get_global_id(0),n=d[1],r=i/512,c=i%512;if(r<2*n)y[i]=r<n?a[r*512+c]:b[(r-n)*512+c];
}
__kernel void merge_branches(__global const ACT*x,__global ACT*y,int R,__global const int*d){
 int i=get_global_id(0);if(i<d[1]*512)y[i]=x[i]+x[d[1]*512+i];
}
