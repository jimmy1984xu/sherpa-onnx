// GPU work-groups execute model, beam and context transitions. No intermediate
// logits, recurrent tensors or candidates cross the host boundary.
#define MAX_BEAM 64
#define MAX_STEPS 4128
#define MAX_TOKENS 3750
#define SCORE_OFFSET 176104
typedef struct {
  float score;
  int frame, on_frame, token, update, context, path, count, slot;
  int emit, duration;
  float acoustic;
} BeamHyp;
typedef struct { int parent, token, frame, duration; float acoustic; } PathNode;

typedef struct { int size, frame, nodes, rounds, finished, evaluations, max_active, reserved; } BeamControl;
__attribute__((reqd_work_group_size(WG,1,1))) __kernel void beam_init(
 __global int *out,__global const int *meta,__global float *states,
 __global BeamHyp *active,__global BeamControl *control,__global const half *encoder,int beam) {
 int l=get_local_id(0);
 for(int i=l;i<5*H;i+=WG)states[i]=0;
 if(!l){out[0]=0;out[1]=0;BeamHyp first={0,0,0,VOCAB-1,1,0,-1,0,0,0,0,0};active[0]=first;
 BeamControl initial={1,0,0,0,0,0,1,0};*control=initial;}
 barrier(CLK_GLOBAL_MEM_FENCE);
 for(int i=l;i<meta[2]*H;i+=WG)if(!isfinite(convert_float(encoder[i])))atomic_or(out,2);
 barrier(CLK_GLOBAL_MEM_FENCE);
 if(!l && out[0])control->finished=1;
}
inline int beam_enabled(__global const BeamHyp *active,__global const BeamControl *control,int row) {
 return !control->finished && row<control->size && active[row].frame==control->frame && active[row].on_frame<10;
}
__attribute__((reqd_work_group_size(WG,1,1))) __kernel void beam_prepare(
 __global float *states,__global const BeamHyp *active,__global BeamControl *control,
 __global char4 *q,__global float *scales,__global int *zps,int beam) {
 int row=get_group_id(0),l=get_local_id(0);if(!beam_enabled(active,control,row))return;
 for(int i=l;i<5*H;i+=WG)states[(beam+row)*5*H+i]=states[row*5*H+i];
 if(!l)atomic_inc(&control->evaluations);
 if(!active[row].update)return;
 __local float x[H],lo[WG],hi[WG],scale[1];__local int zp[1];__local char4 packed[160];
 for(int bank=0;bank<2;++bank){
  for(int i=l;i<H;i+=WG)x[i]=states[row*5*H+bank*H+i];
  barrier(CLK_LOCAL_MEM_FENCE);quant(x,packed,scale,zp,lo,hi);
  for(int i=l;i<160;i+=WG)q[(bank*beam+row)*160+i]=packed[i];
  if(!l){scales[bank*beam+row]=scale[0];zps[bank*beam+row]=zp[0];}
 }
}
__attribute__((reqd_work_group_size(WG,1,1))) __kernel void beam_cell(
 __global float *states,__global const BeamHyp *active,__global const BeamControl *control,
 __global const float *gates,__global char4 *q,__global float *scales,__global int *zps,
 int beam,int bank) {
 int row=get_group_id(0),l=get_local_id(0);if(!beam_enabled(active,control,row)||!active[row].update)return;
 __local float x[H],lo[WG],hi[WG],scale[1];__local int zp[1];__local char4 packed[160];
 int base=(beam+row)*5*H,gate=(bank*beam+row)*G;
 for(int i=l;i<H;i+=WG){float c=sigmoid(gates[gate+H+i])*states[base+(bank+2)*H+i]+sigmoid(gates[gate+i])*tanh(gates[gate+2*H+i]);
  states[base+(bank+2)*H+i]=c;x[i]=sigmoid(gates[gate+3*H+i])*tanh(c);states[base+bank*H+i]=x[i];}
 barrier(CLK_LOCAL_MEM_FENCE);quant(x,packed,scale,zp,lo,hi);
 for(int i=l;i<160;i+=WG)q[(bank*beam+row)*160+i]=packed[i];
 if(!l){scales[bank*beam+row]=scale[0];zps[bank*beam+row]=zp[0];}
}
__attribute__((reqd_work_group_size(WG,1,1))) __kernel void beam_joint(
 __global const half *encoder,__global const float *states,__global const BeamHyp *active,
 __global const BeamControl *control,__global char4 *q,__global float *scales,__global int *zps,int beam) {
 int row=get_group_id(0),l=get_local_id(0);if(!beam_enabled(active,control,row))return;
 __local float x[H],lo[WG],hi[WG],scale[1];__local int zp[1];__local char4 packed[160];
 for(int i=l;i<H;i+=WG)x[i]=fmax(convert_float(encoder[control->frame*H+i])+states[(beam+row)*5*H+4*H+i],0.0f);
 barrier(CLK_LOCAL_MEM_FENCE);quant(x,packed,scale,zp,lo,hi);
 for(int i=l;i<160;i+=WG)q[(2*beam+row)*160+i]=packed[i];
 if(!l){scales[2*beam+row]=scale[0];zps[2*beam+row]=zp[0];}
}
#pragma OPENCL EXTENSION cl_arm_matrix_multiply : enable
// 4x4x16 cooperative layout follows the encoder's existing Arm-derived MMUL.
__attribute__((reqd_work_group_size(64,1,1))) __kernel void beam_affine(
 __global const uchar *w,__global const char *quantized,__global const float *scales,
 __global const int *zps,__global const BeamHyp *active,__global const BeamControl *control,
 __global float *states,__global float *gates,__global float *all_logits,__global int *out,
 int beam,int operation,int trace) {
 if(control->finished)return;
 // A whole MMUL group has the same four rows. Skip matrices for rounds
 // where none of those rows needs this affine transform.
 int row_base=get_global_id(1)*4,needed=0;
 for(int r=0;r<4;++r)needed|=beam_enabled(active,control,row_base+r) &&
   (operation==3 || active[row_base+r].update);
 if(!needed)return;
 int lane=get_global_id(0)&15,x=lane&3,y=lane>>2;
 int row=get_global_id(1)*4+y,column=(get_global_id(0)>>4)*16+x*4;
 int second=operation==0?get_global_id(2):0;
 int enabled=beam_enabled(active,control,row) && (operation==3 || active[row].update);
 ulong W,S,B,SUM;int N,PAD,bank;
 if(operation==0){W=second?H1_W:H0_W;S=second?H1_S:H0_S;B=second?H1_B:H0_B;SUM=second?H1_SUM:H0_SUM;N=PAD=G;bank=second;}
 else if(operation==1){W=X1_W;S=X1_S;B=X1_B;SUM=X1_SUM;N=PAD=G;bank=0;}
 else if(operation==2){W=PRED_W;S=PRED_S;B=PRED_B;SUM=PRED_SUM;N=PAD=H;bank=1;}
 else {W=OUT_W;S=OUT_S;B=OUT_B;SUM=OUT_SUM;N=VOCAB+5;PAD=OUT_PAD;bank=2;}
 int acc[4]={0,0,0,0};
 for(int k=0;k<H;k+=16){
  char4 lhs=enabled?vload4(0,quantized+(bank*beam+row)*H+k+x*4):(char4)(0);
  for(int n=0;n<4;++n){char4 rhs=column+n<PAD?vload4(0,(__global const char *)(w+W)+((k/16)*PAD+column+n)*16+y*4):(char4)(0);
   acc[n]=arm_matrix_multiply(lhs,rhs,acc[n]);}
 }
 if(!enabled)return;
 for(int n=0;n<4;++n)if(column+n<N){
  int o=column+n,value=acc[n]+(128-zps[bank*beam+row])*pi(w,SUM,o);
  float result=convert_float(value)*(scales[bank*beam+row]*pf(w,S,o))+pf(w,B,o);
  if(operation==0){if(!second)result=pf(w,EMBED_GATES,active[row].token*G+o)+result;gates[(second*beam+row)*G+o]=result;}
  else if(operation==1)gates[(beam+row)*G+o]=result+gates[(beam+row)*G+o];
  else if(operation==2)states[(beam+row)*5*H+4*H+o]=result;
  else {all_logits[row*(2*VOCAB+5)+o]=result;if(trace&&control->rounds==0&&row==0&&o<VOCAB)((__global float *)((__global uchar *)out+191104))[o]=result;}
  if(!isfinite(result))atomic_or(out,4);
 }
}
// Each candidate gets its own work-group so independent vocabularies use
// multiple GPU cores; only the following small kernel performs global pruning.
__attribute__((reqd_work_group_size(512,1,1))) __kernel void beam_expand(
 __global const BeamHyp *active,__global BeamHyp *candidates,
 __global float *all_logits,__global const int *graph,__global const BeamControl *control,
 int beam,float blank_penalty) {
 int a=get_group_id(0),l=get_local_id(0);
 if(control->finished || a>=control->size)return;
 BeamHyp hyp=active[a];
 for(int k=l;k<beam+1;k+=512)candidates[a*(beam+1)+k].score=-INFINITY;
 if(hyp.frame>control->frame || hyp.on_frame>=10){
  if(!l){if(hyp.on_frame>=10 && hyp.frame==control->frame){++hyp.frame;hyp.on_frame=0;}
   hyp.emit=0;hyp.slot=a;candidates[a*(beam+1)]=hyp;}
  return;
 }
 __local int duration,indices[512];
 __local float best[512],sums[512],normalizer,duration_score;
 int stride=graph[1]+1,table_count=graph[0]*stride;
 __global const int *map=graph+4,*next=map+VOCAB;
 __global const float *bonus=(__global const float *)(next+table_count);
  __global float *logits=all_logits+a*(2*VOCAB+5);
      float maximum=-INFINITY;
      for(int o=l;o<VOCAB;o+=512) maximum=fmax(maximum,logits[o]);
      best[l]=maximum;
      barrier(CLK_LOCAL_MEM_FENCE);
      for(int n=512/2;n;n/=2) {
        if(l<n) best[l]=fmax(best[l],best[l+n]);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      float sum=0;
      for(int o=l;o<VOCAB;o+=512) sum+=exp(logits[o]-best[0]);
      sums[l]=sum;
      barrier(CLK_LOCAL_MEM_FENCE);
      for(int n=512/2;n;n/=2) {
        if(l<n) sums[l]+=sums[l+n];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if(!l) {
        normalizer=best[0]+log(sums[0]); duration=0;
        for(int i=1;i<5;++i) if(logits[VOCAB+i]>logits[VOCAB+duration]) duration=i;
        float total=0,maximum=logits[VOCAB+duration];
        for(int i=0;i<5;++i) total+=exp(logits[VOCAB+i]-maximum);
        duration_score=-log(total);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // The first VOCAB entries hold unbiased log probabilities; the second
      // bank is scratch for context-boosted top-k ranking.
      for(int o=l;o<VOCAB;o+=512) {
        float acoustic=logits[o]-normalizer;
        int transition=hyp.context*stride+map[o];
        float reward=o==VOCAB-1?-blank_penalty:bonus[transition];
        logits[o]=acoustic; logits[VOCAB+5+o]=acoustic+reward;
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
      for(int k=0;k<beam;++k) {
        float value=-INFINITY;int index=2147483647;
        for(int o=l;o<VOCAB;o+=512) {
          float v=logits[VOCAB+5+o];
          if(v>value || (v==value && o<index)) { value=v;index=o; }
        }
        best[l]=value;indices[l]=index;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int n=512/2;n;n/=2) {
          if(l<n && (best[l+n]>best[l] || (best[l+n]==best[l] && indices[l+n]<indices[l]))) {
            best[l]=best[l+n];indices[l]=indices[l+n];
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(!l) {
          int y=indices[0];BeamHyp child=hyp;
          child.score+=best[0]+duration_score;child.slot=beam+a;
          child.duration=duration;child.emit=y==VOCAB-1?0:y+1;child.acoustic=logits[y];
          if(y==VOCAB-1) { child.frame+=max(1,duration);child.on_frame=0;child.update=0; }
          else {
            child.frame+=duration;child.on_frame=duration?0:hyp.on_frame+1;
            child.token=y;child.update=1;child.context=next[hyp.context*stride+map[y]];
            ++child.count;
          }
          candidates[a*(beam+1)+k]=child;
          logits[VOCAB+5+y]=-INFINITY;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
}
__attribute__((reqd_work_group_size(128,1,1))) __kernel void beam_select(
 __global int *out,__global const int *meta,__global float *states,
 __global BeamHyp *active,__global BeamHyp *candidates,__global PathNode *paths,
 __global BeamControl *control,int beam) {
 int l=get_local_id(0),frames=meta[2];if(control->finished)return;
 __local int selection[MAX_BEAM],size,nodes;
  if(!l){nodes=control->nodes;int count=control->size*(beam+1);size=min(beam,count);
   for(int k=0;k<size;++k){int selected=-1;float value=-INFINITY;
    for(int i=0;i<count;++i){int used=0;for(int j=0;j<k;++j)used|=selection[j]==i;
     if(!used&&(selected<0||candidates[i].score>value)){selected=i;value=candidates[i].score;}}
    if(!isfinite(value)){size=k;break;}selection[k]=selected;
   }
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  for(int k=0;k<size;++k){int source=candidates[selection[k]].slot;for(int i=l;i<5*H;i+=128)states[(2*beam+k)*5*H+i]=states[source*5*H+i];}
  barrier(CLK_GLOBAL_MEM_FENCE);
  for(int k=0;k<size;++k){for(int i=l;i<5*H;i+=128)states[k*5*H+i]=states[(2*beam+k)*5*H+i];
   if(!l){BeamHyp h=candidates[selection[k]];if(h.count>MAX_TOKENS)atomic_or(out,8);
    if(h.emit){PathNode n={h.path,h.emit-1,h.frame-h.duration,h.duration,h.acoustic};paths[nodes]=n;h.path=nodes++;}
    h.slot=k;h.emit=0;active[k]=h;
   }}
  barrier(CLK_GLOBAL_MEM_FENCE);
  if(!l){control->size=size;control->nodes=nodes;++control->rounds;control->max_active=max(control->max_active,size);
   int minimum=frames;for(int i=0;i<size;++i)minimum=min(minimum,active[i].frame);control->frame=minimum;
   control->finished=minimum>=frames||out[0];if(!size){atomic_or(out,32);control->finished=1;}
   if(control->rounds>=MAX_STEPS&&!control->finished){atomic_or(out,16);control->finished=1;}}
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}
// Four candidate rows share matrix loads and MMUL instructions inside a single
// persistent work-group. Quantization uses four independent 64-lane reductions.
inline void fast_quant(__local float *x,__local char4 *q,__local float *scales,
 __local int *zps,__local float *lo,__local float *hi,int bank) {
 int l=get_local_id(0),group=l/64,lane=l%64;
 float a=0,b=0;
 for(int i=lane;i<H;i+=64){a=fmin(a,x[group*H+i]);b=fmax(b,x[group*H+i]);}
 lo[l]=a;hi[l]=b;barrier(CLK_LOCAL_MEM_FENCE);
 for(int n=32;n;n/=2){if(lane<n){lo[l]=fmin(lo[l],lo[l+n]);hi[l]=fmax(hi[l],hi[l+n]);}barrier(CLK_LOCAL_MEM_FENCE);}
 if(!lane){float value=(hi[l]-lo[l])/255.0f;scales[bank*4+group]=value==0?1:value;zps[bank*4+group]=clamp(convert_int_rte(-lo[l]/scales[bank*4+group]),0,255);}
 barrier(CLK_LOCAL_MEM_FENCE);
 for(int i=lane;i<160;i+=64){int4 value=convert_int4_rte(vload4(i,x+group*H)/scales[bank*4+group])+(int4)(zps[bank*4+group]);q[(bank*4+group)*160+i]=convert_char4(clamp(value,(int4)(0),(int4)(255))-(int4)(128));}
 barrier(CLK_LOCAL_MEM_FENCE);
}
inline void fast_affine(__global const uchar *w,__local const char4 *q,
 __local const float *scales,__local const int *zps,__global const BeamHyp *active,
 __global const BeamControl *control,__global float *states,__global float *gates,
 __global float *all_logits,__global int *out,int beam,int base,int operation,int trace) {
 int l=get_local_id(0),lane=l&15,tx=lane&3,ty=lane>>2,row=base+ty;
 int enabled=beam_enabled(active,control,row) && (operation==3 || active[row].update);
 int passes=operation==0?2:1;
 for(int second=0;second<passes;++second) {
  ulong W,S,B,SUM;int N,PAD,bank;
  if(operation==0){W=second?H1_W:H0_W;S=second?H1_S:H0_S;B=second?H1_B:H0_B;SUM=second?H1_SUM:H0_SUM;N=PAD=G;bank=second;}
  else if(operation==1){W=X1_W;S=X1_S;B=X1_B;SUM=X1_SUM;N=PAD=G;bank=0;}
  else if(operation==2){W=PRED_W;S=PRED_S;B=PRED_B;SUM=PRED_SUM;N=PAD=H;bank=1;}
  else {W=OUT_W;S=OUT_S;B=OUT_B;SUM=OUT_SUM;N=VOCAB+5;PAD=OUT_PAD;bank=2;}
  for(int tile=0;tile<N;tile+=256) {
   int column=tile+(l>>4)*16+tx*4,acc[4]={0,0,0,0};
   for(int k=0;k<H;k+=16){
    char4 lhs=enabled?vload4(0,(__local const char *)q+(bank*4+ty)*H+k+tx*4):(char4)(0);
    for(int n=0;n<4;++n){char4 rhs=column+n<PAD?vload4(0,(__global const char *)(w+W)+((k/16)*PAD+column+n)*16+ty*4):(char4)(0);acc[n]=arm_matrix_multiply(lhs,rhs,acc[n]);}
   }
   if(enabled)for(int n=0;n<4;++n)if(column+n<N){
    int o=column+n,value=acc[n]+(128-zps[bank*4+ty])*pi(w,SUM,o);
    float result=convert_float(value)*(scales[bank*4+ty]*pf(w,S,o))+pf(w,B,o);
    if(operation==0){if(!second)result=pf(w,EMBED_GATES,active[row].token*G+o)+result;gates[(second*beam+row)*G+o]=result;}
    else if(operation==1)gates[(beam+row)*G+o]=result+gates[(beam+row)*G+o];
    else if(operation==2)states[(beam+row)*5*H+4*H+o]=result;
    else {all_logits[row*(2*VOCAB+5)+o]=result;if(trace&&control->rounds==0&&row==0&&o<VOCAB)((__global float *)((__global uchar *)out+191104))[o]=result;}
    if(!isfinite(result))atomic_or(out,4);
   }
  }
 }
 barrier(CLK_GLOBAL_MEM_FENCE);
}
__attribute__((reqd_work_group_size(256,1,1))) __kernel void beam_fast(
 __global const half *encoder,__global const uchar *w,__global int *out,__global const int *meta,
 __global float *states,__global BeamHyp *active,__global BeamHyp *candidates,
 __global PathNode *paths,__global float *all_logits,__global const int *graph,
 __global BeamControl *control,__global float *gates,int beam,float blank_penalty,int trace) {
 int l=get_local_id(0),group=l/64,lane=l%64,frames=meta[2];
 __local float x[4*H],lo[256],hi[256],scales[12],best[256],sums[256],normalizer[4],duration_score[4];
 __local char4 packed[12*160];__local int zps[12],indices[256],duration[4],selection[MAX_BEAM],size,nodes,any;
 int stride=graph[1]+1,table_count=graph[0]*stride;
 __global const int *map=graph+4,*next=map+VOCAB;
 __global const float *bonus=(__global const float *)(next+table_count),*finalize=bonus+table_count;
 while(!control->finished) {
  for(int base=0;base<control->size;base+=4) {
   int row=base+group,enabled=beam_enabled(active,control,row);
   if(!l){any=0;for(int i=0;i<4;++i)any|=beam_enabled(active,control,base+i);}
   barrier(CLK_LOCAL_MEM_FENCE);if(!any)continue;
   if(enabled)for(int i=lane;i<5*H;i+=64)states[(beam+row)*5*H+i]=states[row*5*H+i];
   for(int bank=0;bank<2;++bank){
    for(int i=lane;i<H;i+=64)x[group*H+i]=enabled?states[row*5*H+bank*H+i]:0;
    barrier(CLK_LOCAL_MEM_FENCE);fast_quant(x,packed,scales,zps,lo,hi,bank);
   }
   fast_affine(w,packed,scales,zps,active,control,states,gates,all_logits,out,beam,base,0,trace);
   for(int bank=0;bank<2;++bank) {
    for(int i=lane;i<H;i+=64){
     float value=0;
     if(enabled){int state=(beam+row)*5*H,gate=(bank*beam+row)*G;
      if(active[row].update){float c=sigmoid(gates[gate+H+i])*states[state+(bank+2)*H+i]+sigmoid(gates[gate+i])*tanh(gates[gate+2*H+i]);
       states[state+(bank+2)*H+i]=c;value=sigmoid(gates[gate+3*H+i])*tanh(c);states[state+bank*H+i]=value;}
      else value=states[state+bank*H+i];
     }
     x[group*H+i]=value;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);fast_quant(x,packed,scales,zps,lo,hi,bank);
    fast_affine(w,packed,scales,zps,active,control,states,gates,all_logits,out,beam,base,bank==0?1:2,trace);
   }
   for(int i=lane;i<H;i+=64)x[group*H+i]=enabled?fmax(convert_float(encoder[control->frame*H+i])+states[(beam+row)*5*H+4*H+i],0.0f):0;
   barrier(CLK_LOCAL_MEM_FENCE);fast_quant(x,packed,scales,zps,lo,hi,2);
   fast_affine(w,packed,scales,zps,active,control,states,gates,all_logits,out,beam,base,3,trace);
   if(!lane&&enabled)atomic_inc(&control->evaluations);
  }
  // Four independent 64-lane context/top-k reductions execute in parallel.
  for(int base=0;base<control->size;base+=4) {
   int row=base+group,valid=row<control->size,enabled=beam_enabled(active,control,row);
   BeamHyp h;if(valid)h=active[row];
   for(int k=lane;k<beam+1;k+=64)if(valid)candidates[row*(beam+1)+k].score=-INFINITY;
   if(valid&&!enabled&&!lane){if(h.frame==control->frame&&h.on_frame>=10){++h.frame;h.on_frame=0;}h.emit=0;h.slot=row;candidates[row*(beam+1)]=h;}
   __global float *logits=all_logits+(valid?row:0)*(2*VOCAB+5);
   float maximum=-INFINITY;
   for(int o=lane;o<VOCAB;o+=64)maximum=fmax(maximum,enabled?logits[o]:0);
   best[l]=maximum;barrier(CLK_LOCAL_MEM_FENCE);
   for(int n=32;n;n/=2){if(lane<n)best[l]=fmax(best[l],best[l+n]);barrier(CLK_LOCAL_MEM_FENCE);}
   float sum=0;for(int o=lane;o<VOCAB;o+=64)sum+=exp((enabled?logits[o]:0)-best[group*64]);
   sums[l]=sum;barrier(CLK_LOCAL_MEM_FENCE);
   for(int n=32;n;n/=2){if(lane<n)sums[l]+=sums[l+n];barrier(CLK_LOCAL_MEM_FENCE);}
   if(!lane){normalizer[group]=best[group*64]+log(sums[group*64]);duration[group]=0;
    if(enabled){for(int i=1;i<5;++i)if(logits[VOCAB+i]>logits[VOCAB+duration[group]])duration[group]=i;
     float total=0,m=logits[VOCAB+duration[group]];for(int i=0;i<5;++i)total+=exp(logits[VOCAB+i]-m);duration_score[group]=-log(total);}}
   barrier(CLK_LOCAL_MEM_FENCE);
   if(enabled)for(int o=lane;o<VOCAB;o+=64){float acoustic=logits[o]-normalizer[group];logits[o]=acoustic;logits[VOCAB+5+o]=acoustic+(o==VOCAB-1?-blank_penalty:bonus[h.context*stride+map[o]]);}
   barrier(CLK_GLOBAL_MEM_FENCE);
   for(int k=0;k<beam;++k){
    float value=-INFINITY;int index=2147483647;
    if(enabled)for(int o=lane;o<VOCAB;o+=64){float v=logits[VOCAB+5+o];if(v>value||(v==value&&o<index)){value=v;index=o;}}
    best[l]=value;indices[l]=index;barrier(CLK_LOCAL_MEM_FENCE);
    for(int n=32;n;n/=2){if(lane<n&&(best[l+n]>best[l]||(best[l+n]==best[l]&&indices[l+n]<indices[l]))){best[l]=best[l+n];indices[l]=indices[l+n];}barrier(CLK_LOCAL_MEM_FENCE);}
    if(!lane&&enabled){int y=indices[group*64],d=duration[group];BeamHyp child=h;
     child.score+=best[group*64]+duration_score[group];child.slot=beam+row;child.duration=d;child.emit=y==VOCAB-1?0:y+1;child.acoustic=logits[y];
     if(y==VOCAB-1){child.frame+=max(1,d);child.on_frame=0;child.update=0;}
     else {child.frame+=d;child.on_frame=d?0:h.on_frame+1;child.token=y;child.update=1;child.context=next[h.context*stride+map[y]];++child.count;}
     candidates[row*(beam+1)+k]=child;logits[VOCAB+5+y]=-INFINITY;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
   }
  }
  if(!l){nodes=control->nodes;int count=control->size*(beam+1);size=min(beam,count);
   for(int k=0;k<size;++k){int selected=-1;float value=-INFINITY;
    for(int i=0;i<count;++i){int used=0;for(int j=0;j<k;++j)used|=selection[j]==i;
     if(!used&&(selected<0||candidates[i].score>value)){selected=i;value=candidates[i].score;}}
    if(!isfinite(value)){size=k;break;}selection[k]=selected;
   }
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  for(int k=0;k<size;++k){int source=candidates[selection[k]].slot;for(int i=l;i<5*H;i+=256)states[(2*beam+k)*5*H+i]=states[source*5*H+i];}
  barrier(CLK_GLOBAL_MEM_FENCE);
  for(int k=0;k<size;++k){for(int i=l;i<5*H;i+=256)states[k*5*H+i]=states[(2*beam+k)*5*H+i];
   if(!l){BeamHyp h=candidates[selection[k]];if(h.count>MAX_TOKENS)atomic_or(out,8);
    if(h.emit){PathNode n={h.path,h.emit-1,h.frame-h.duration,h.duration,h.acoustic};paths[nodes]=n;h.path=nodes++;}
    h.slot=k;h.emit=0;active[k]=h;
   }}
  barrier(CLK_GLOBAL_MEM_FENCE);
  if(!l){control->size=size;control->nodes=nodes;++control->rounds;control->max_active=max(control->max_active,size);
   int minimum=frames;for(int i=0;i<size;++i)minimum=min(minimum,active[i].frame);control->frame=minimum;
   control->finished=minimum>=frames||out[0];if(!size){atomic_or(out,32);control->finished=1;}
   if(control->rounds>=MAX_STEPS&&!control->finished){atomic_or(out,16);control->finished=1;}}
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
 }
 if(!l){int selected=0;float best_score=-INFINITY;
  for(int i=0;i<control->size;++i){float score=active[i].score+finalize[active[i].context];if(score>best_score){selected=i;best_score=score;}}
  BeamHyp h=active[selected];out[1]=min(h.count,MAX_TOKENS);out[2]=control->rounds;out[3]=(int)(((uint)control->max_active<<24)|(uint)control->evaluations);
  __global float *scores=(__global float *)((__global uchar *)out+SCORE_OFFSET);int path=h.path;
  for(int i=out[1]-1;i>=0;--i){PathNode n=paths[path];out[8+3*i]=n.token;out[9+3*i]=n.frame;out[10+3*i]=n.duration;scores[i]=n.acoustic;path=n.parent;}
 }
}
