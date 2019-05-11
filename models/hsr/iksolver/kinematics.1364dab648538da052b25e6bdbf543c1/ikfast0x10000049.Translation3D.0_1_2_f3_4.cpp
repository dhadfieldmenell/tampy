/// autogenerated analytical inverse kinematics code from ikfast program part of OpenRAVE
/// \author Rosen Diankov
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///     http://www.apache.org/licenses/LICENSE-2.0
/// 
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
///
/// ikfast version 0x10000049 generated on 2018-12-29 11:02:16.633839
/// To compile with gcc:
///     gcc -lstdc++ ik.cpp
/// To compile without any main function as a shared object (might need -llapack):
///     gcc -fPIC -lstdc++ -DIKFAST_NO_MAIN -DIKFAST_CLIBRARY -shared -Wl,-soname,libik.so -o libik.so ik.cpp
#define IKFAST_HAS_LIBRARY
#include "ikfast.h" // found inside share/openrave-X.Y/python/ikfast.h
using namespace ikfast;

// check if the included ikfast version matches what this file was compiled with
#define IKFAST_COMPILE_ASSERT(x) extern int __dummy[(int)x]
IKFAST_COMPILE_ASSERT(IKFAST_VERSION==0x10000049);

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <complex>

#ifndef IKFAST_ASSERT
#include <stdexcept>
#include <sstream>
#include <iostream>

#ifdef _MSC_VER
#ifndef __PRETTY_FUNCTION__
#define __PRETTY_FUNCTION__ __FUNCDNAME__
#endif
#endif

#ifndef __PRETTY_FUNCTION__
#define __PRETTY_FUNCTION__ __func__
#endif

#define IKFAST_ASSERT(b) { if( !(b) ) { std::stringstream ss; ss << "ikfast exception: " << __FILE__ << ":" << __LINE__ << ": " <<__PRETTY_FUNCTION__ << ": Assertion '" << #b << "' failed"; throw std::runtime_error(ss.str()); } }

#endif

#if defined(_MSC_VER)
#define IKFAST_ALIGNED16(x) __declspec(align(16)) x
#else
#define IKFAST_ALIGNED16(x) x __attribute((aligned(16)))
#endif

#define IK2PI  ((IkReal)6.28318530717959)
#define IKPI  ((IkReal)3.14159265358979)
#define IKPI_2  ((IkReal)1.57079632679490)

#ifdef _MSC_VER
#ifndef isnan
#define isnan _isnan
#endif
#ifndef isinf
#define isinf _isinf
#endif
//#ifndef isfinite
//#define isfinite _isfinite
//#endif
#endif // _MSC_VER

// lapack routines
extern "C" {
  void dgetrf_ (const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
  void zgetrf_ (const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info);
  void dgetri_(const int* n, const double* a, const int* lda, int* ipiv, double* work, const int* lwork, int* info);
  void dgesv_ (const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info);
  void dgetrs_(const char *trans, const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, double *b, const int *ldb, int *info);
  void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *wr, double *wi,double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
}

using namespace std; // necessary to get std math routines

#ifdef IKFAST_NAMESPACE
namespace IKFAST_NAMESPACE {
#endif

inline float IKabs(float f) { return fabsf(f); }
inline double IKabs(double f) { return fabs(f); }

inline float IKsqr(float f) { return f*f; }
inline double IKsqr(double f) { return f*f; }

inline float IKlog(float f) { return logf(f); }
inline double IKlog(double f) { return log(f); }

// allows asin and acos to exceed 1. has to be smaller than thresholds used for branch conds and evaluation
#ifndef IKFAST_SINCOS_THRESH
#define IKFAST_SINCOS_THRESH ((IkReal)1e-7)
#endif

// used to check input to atan2 for degenerate cases. has to be smaller than thresholds used for branch conds and evaluation
#ifndef IKFAST_ATAN2_MAGTHRESH
#define IKFAST_ATAN2_MAGTHRESH ((IkReal)1e-7)
#endif

// minimum distance of separate solutions
#ifndef IKFAST_SOLUTION_THRESH
#define IKFAST_SOLUTION_THRESH ((IkReal)1e-6)
#endif

// there are checkpoints in ikfast that are evaluated to make sure they are 0. This threshold speicfies by how much they can deviate
#ifndef IKFAST_EVALCOND_THRESH
#define IKFAST_EVALCOND_THRESH ((IkReal)0.00001)
#endif


inline float IKasin(float f)
{
IKFAST_ASSERT( f > -1-IKFAST_SINCOS_THRESH && f < 1+IKFAST_SINCOS_THRESH ); // any more error implies something is wrong with the solver
if( f <= -1 ) return float(-IKPI_2);
else if( f >= 1 ) return float(IKPI_2);
return asinf(f);
}
inline double IKasin(double f)
{
IKFAST_ASSERT( f > -1-IKFAST_SINCOS_THRESH && f < 1+IKFAST_SINCOS_THRESH ); // any more error implies something is wrong with the solver
if( f <= -1 ) return -IKPI_2;
else if( f >= 1 ) return IKPI_2;
return asin(f);
}

// return positive value in [0,y)
inline float IKfmod(float x, float y)
{
    while(x < 0) {
        x += y;
    }
    return fmodf(x,y);
}

// return positive value in [0,y)
inline double IKfmod(double x, double y)
{
    while(x < 0) {
        x += y;
    }
    return fmod(x,y);
}

inline float IKacos(float f)
{
IKFAST_ASSERT( f > -1-IKFAST_SINCOS_THRESH && f < 1+IKFAST_SINCOS_THRESH ); // any more error implies something is wrong with the solver
if( f <= -1 ) return float(IKPI);
else if( f >= 1 ) return float(0);
return acosf(f);
}
inline double IKacos(double f)
{
IKFAST_ASSERT( f > -1-IKFAST_SINCOS_THRESH && f < 1+IKFAST_SINCOS_THRESH ); // any more error implies something is wrong with the solver
if( f <= -1 ) return IKPI;
else if( f >= 1 ) return 0;
return acos(f);
}
inline float IKsin(float f) { return sinf(f); }
inline double IKsin(double f) { return sin(f); }
inline float IKcos(float f) { return cosf(f); }
inline double IKcos(double f) { return cos(f); }
inline float IKtan(float f) { return tanf(f); }
inline double IKtan(double f) { return tan(f); }
inline float IKsqrt(float f) { if( f <= 0.0f ) return 0.0f; return sqrtf(f); }
inline double IKsqrt(double f) { if( f <= 0.0 ) return 0.0; return sqrt(f); }
inline float IKatan2Simple(float fy, float fx) {
    return atan2f(fy,fx);
}
inline float IKatan2(float fy, float fx) {
    if( isnan(fy) ) {
        IKFAST_ASSERT(!isnan(fx)); // if both are nan, probably wrong value will be returned
        return float(IKPI_2);
    }
    else if( isnan(fx) ) {
        return 0;
    }
    return atan2f(fy,fx);
}
inline double IKatan2Simple(double fy, double fx) {
    return atan2(fy,fx);
}
inline double IKatan2(double fy, double fx) {
    if( isnan(fy) ) {
        IKFAST_ASSERT(!isnan(fx)); // if both are nan, probably wrong value will be returned
        return IKPI_2;
    }
    else if( isnan(fx) ) {
        return 0;
    }
    return atan2(fy,fx);
}

template <typename T>
struct CheckValue
{
    T value;
    bool valid;
};

template <typename T>
inline CheckValue<T> IKatan2WithCheck(T fy, T fx, T epsilon)
{
    CheckValue<T> ret;
    ret.valid = false;
    ret.value = 0;
    if( !isnan(fy) && !isnan(fx) ) {
        if( IKabs(fy) >= IKFAST_ATAN2_MAGTHRESH || IKabs(fx) > IKFAST_ATAN2_MAGTHRESH ) {
            ret.value = IKatan2Simple(fy,fx);
            ret.valid = true;
        }
    }
    return ret;
}

inline float IKsign(float f) {
    if( f > 0 ) {
        return float(1);
    }
    else if( f < 0 ) {
        return float(-1);
    }
    return 0;
}

inline double IKsign(double f) {
    if( f > 0 ) {
        return 1.0;
    }
    else if( f < 0 ) {
        return -1.0;
    }
    return 0;
}

template <typename T>
inline CheckValue<T> IKPowWithIntegerCheck(T f, int n)
{
    CheckValue<T> ret;
    ret.valid = true;
    if( n == 0 ) {
        ret.value = 1.0;
        return ret;
    }
    else if( n == 1 )
    {
        ret.value = f;
        return ret;
    }
    else if( n < 0 )
    {
        if( f == 0 )
        {
            ret.valid = false;
            ret.value = (T)1.0e30;
            return ret;
        }
        if( n == -1 ) {
            ret.value = T(1.0)/f;
            return ret;
        }
    }

    int num = n > 0 ? n : -n;
    if( num == 2 ) {
        ret.value = f*f;
    }
    else if( num == 3 ) {
        ret.value = f*f*f;
    }
    else {
        ret.value = 1.0;
        while(num>0) {
            if( num & 1 ) {
                ret.value *= f;
            }
            num >>= 1;
            f *= f;
        }
    }
    
    if( n < 0 ) {
        ret.value = T(1.0)/ret.value;
    }
    return ret;
}

/// solves the forward kinematics equations.
/// \param pfree is an array specifying the free joints of the chain.
IKFAST_API void ComputeFk(const IkReal* j, IkReal* eetrans, IkReal* eerot) {
IkReal x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13;
x0=IKcos(j[1]);
x1=IKsin(j[1]);
x2=IKcos(j[2]);
x3=IKsin(j[3]);
x4=IKcos(j[3]);
x5=IKsin(j[4]);
x6=IKcos(j[4]);
x7=IKsin(j[2]);
x8=((0.2732)*x0);
x9=((0.012)*x4);
x10=((0.012)*x5);
x11=((0.2732)*x1);
x12=((0.012)*x3);
x13=(x2*x3);
eetrans[0]=((0.141)+(((-1.0)*x11*x4))+(((0.005)*x0))+((x6*((((x0*x2*x9))+(((-1.0)*x1*x12))))))+(((-1.0)*x0*x10*x7))+(((-0.345)*x1))+(((-1.0)*x13*x8)));
eetrans[1]=((0.078)+(((-0.2732)*x3*x7))+((x6*x7*x9))+((x10*x2)));
eetrans[2]=((0.34)+(((0.345)*x0))+(((-1.0)*x1*x10*x7))+(((-1.0)*x11*x13))+((x4*x8))+(((0.005)*x1))+j[0]+((x6*((((x0*x12))+((x1*x2*x9)))))));
}

IKFAST_API int GetNumFreeParameters() { return 2; }
IKFAST_API int* GetFreeParameters() { static int freeparams[] = {3, 4}; return freeparams; }
IKFAST_API int GetNumJoints() { return 5; }

IKFAST_API int GetIkRealSize() { return sizeof(IkReal); }

IKFAST_API int GetIkType() { return 0x33000003; }

class IKSolver {
public:
IkReal j0,cj0,sj0,htj0,j0mul,j1,cj1,sj1,htj1,j1mul,j2,cj2,sj2,htj2,j2mul,j3,cj3,sj3,htj3,j4,cj4,sj4,htj4,new_px,px,npx,new_py,py,npy,new_pz,pz,npz,pp;
unsigned char _ij0[2], _nj0,_ij1[2], _nj1,_ij2[2], _nj2,_ij3[2], _nj3,_ij4[2], _nj4;

IkReal j100, cj100, sj100;
unsigned char _ij100[2], _nj100;
bool ComputeIk(const IkReal* eetrans, const IkReal* eerot, const IkReal* pfree, IkSolutionListBase<IkReal>& solutions) {
j0=numeric_limits<IkReal>::quiet_NaN(); _ij0[0] = -1; _ij0[1] = -1; _nj0 = -1; j1=numeric_limits<IkReal>::quiet_NaN(); _ij1[0] = -1; _ij1[1] = -1; _nj1 = -1; j2=numeric_limits<IkReal>::quiet_NaN(); _ij2[0] = -1; _ij2[1] = -1; _nj2 = -1;  _ij3[0] = -1; _ij3[1] = -1; _nj3 = 0;  _ij4[0] = -1; _ij4[1] = -1; _nj4 = 0; 
for(int dummyiter = 0; dummyiter < 1; ++dummyiter) {
    solutions.Clear();
j3=pfree[0]; cj3=cos(pfree[0]); sj3=sin(pfree[0]);
j4=pfree[1]; cj4=cos(pfree[1]); sj4=sin(pfree[1]);
px = eetrans[0]; py = eetrans[1]; pz = eetrans[2];

new_px=((-0.141)+px);
new_py=((-0.078)+py);
new_pz=((-0.34)+pz);
px = new_px; py = new_py; pz = new_pz;
pp=((px*px)+(py*py)+(pz*pz));
{
IkReal j2eval[2];
IkReal x14=(cj3*cj4);
j2eval[0]=((x14*x14)+(((-45.5333333333333)*sj3*x14))+(sj4*sj4)+(((518.321111111111)*(sj3*sj3))));
j2eval[1]=((IKabs(sj4))+(((83.3333333333333)*(IKabs(((((0.012)*x14))+(((-0.2732)*sj3))))))));
if( IKabs(j2eval[0]) < 0.0000010000000000  || IKabs(j2eval[1]) < 0.0000010000000000  )
{
continue; // no branches [j0, j1, j2]

} else
{
{
IkReal j2array[2], cj2array[2], sj2array[2];
bool j2valid[2]={false};
_nj2 = 2;
IkReal x15=((((-0.2732)*sj3))+(((0.012)*cj3*cj4)));
CheckValue<IkReal> x18 = IKatan2WithCheck(IkReal(((0.012)*sj4)),IkReal(x15),IKFAST_ATAN2_MAGTHRESH);
if(!x18.valid){
continue;
}
IkReal x16=((1.0)*(x18.value));
if((((x15*x15)+(((0.000144)*(sj4*sj4))))) < -0.00001)
continue;
CheckValue<IkReal> x19=IKPowWithIntegerCheck(IKabs(IKsqrt(((x15*x15)+(((0.000144)*(sj4*sj4)))))),-1);
if(!x19.valid){
continue;
}
if( ((py*(x19.value))) < -1-IKFAST_SINCOS_THRESH || ((py*(x19.value))) > 1+IKFAST_SINCOS_THRESH )
    continue;
IkReal x17=IKasin((py*(x19.value)));
j2array[0]=((((-1.0)*x16))+x17);
sj2array[0]=IKsin(j2array[0]);
cj2array[0]=IKcos(j2array[0]);
j2array[1]=((3.14159265358979)+(((-1.0)*x16))+(((-1.0)*x17)));
sj2array[1]=IKsin(j2array[1]);
cj2array[1]=IKcos(j2array[1]);
if( j2array[0] > IKPI )
{
    j2array[0]-=IK2PI;
}
else if( j2array[0] < -IKPI )
{    j2array[0]+=IK2PI;
}
j2valid[0] = true;
if( j2array[1] > IKPI )
{
    j2array[1]-=IK2PI;
}
else if( j2array[1] < -IKPI )
{    j2array[1]+=IK2PI;
}
j2valid[1] = true;
for(int ij2 = 0; ij2 < 2; ++ij2)
{
if( !j2valid[ij2] )
{
    continue;
}
_ij2[0] = ij2; _ij2[1] = -1;
for(int iij2 = ij2+1; iij2 < 2; ++iij2)
{
if( j2valid[iij2] && IKabs(cj2array[ij2]-cj2array[iij2]) < IKFAST_SOLUTION_THRESH && IKabs(sj2array[ij2]-sj2array[iij2]) < IKFAST_SOLUTION_THRESH )
{
    j2valid[iij2]=false; _ij2[1] = iij2; break; 
}
}
j2 = j2array[ij2]; cj2 = cj2array[ij2]; sj2 = sj2array[ij2];

{
IkReal j0array[2], cj0array[2], sj0array[2];
bool j0valid[2]={false};
_nj0 = 2;
if((((0.19383224)+(((-1.0)*(px*px)))+(((0.00012)*cj2*cj3*cj4))+(((0.188508)*cj3))+(((-0.00012)*sj2*sj4))+(((-0.002732)*cj2*sj3))+(((-1.0)*(py*py)))+(((0.00828)*cj4*sj3)))) < -0.00001)
continue;
IkReal x20=IKsqrt(((0.19383224)+(((-1.0)*(px*px)))+(((0.00012)*cj2*cj3*cj4))+(((0.188508)*cj3))+(((-0.00012)*sj2*sj4))+(((-0.002732)*cj2*sj3))+(((-1.0)*(py*py)))+(((0.00828)*cj4*sj3))));
j0array[0]=(pz+x20);
sj0array[0]=IKsin(j0array[0]);
cj0array[0]=IKcos(j0array[0]);
j0array[1]=((((-1.0)*x20))+pz);
sj0array[1]=IKsin(j0array[1]);
cj0array[1]=IKcos(j0array[1]);
j0valid[0] = true;
j0valid[1] = true;
for(int ij0 = 0; ij0 < 2; ++ij0)
{
if( !j0valid[ij0] )
{
    continue;
}
_ij0[0] = ij0; _ij0[1] = -1;
for(int iij0 = ij0+1; iij0 < 2; ++iij0)
{
if( j0valid[iij0] && IKabs(cj0array[ij0]-cj0array[iij0]) < IKFAST_SOLUTION_THRESH && IKabs(sj0array[ij0]-sj0array[iij0]) < IKFAST_SOLUTION_THRESH )
{
    j0valid[iij0]=false; _ij0[1] = iij0; break; 
}
}
j0 = j0array[ij0]; cj0 = cj0array[ij0]; sj0 = sj0array[ij0];

{
IkReal j1eval[2];
IkReal x21=((((-1.0)*(px*px)))+(((-1.0)*(j0*j0)))+(((2.0)*j0*pz))+(((-1.0)*(pz*pz))));
j1eval[0]=x21;
j1eval[1]=IKsign(x21);
if( IKabs(j1eval[0]) < 0.0000010000000000  || IKabs(j1eval[1]) < 0.0000010000000000  )
{
{
IkReal j1eval[2];
IkReal x22=(cj2*px);
IkReal x23=(cj3*pz);
IkReal x24=((2.4)*cj4);
IkReal x25=(j0*sj3);
IkReal x26=(cj3*j0);
IkReal x27=((0.012)*cj4);
IkReal x28=(pz*sj3);
IkReal x29=(px*sj2*sj4);
j1eval[0]=((((2.4)*x29))+(((54.64)*x23))+(((-69.0)*j0))+(((-1.0)*px))+(((-1.0)*x24*x25))+(((54.64)*sj3*x22))+(((-54.64)*x26))+((x24*x28))+(((-1.0)*cj3*x22*x24))+(((69.0)*pz)));
j1eval[1]=IKsign(((((-0.2732)*x26))+(((0.012)*x29))+(((0.2732)*x23))+(((0.345)*pz))+(((0.2732)*sj3*x22))+(((-1.0)*x25*x27))+(((-0.345)*j0))+(((-0.005)*px))+(((-1.0)*cj3*x22*x27))+((x27*x28))));
if( IKabs(j1eval[0]) < 0.0000010000000000  || IKabs(j1eval[1]) < 0.0000010000000000  )
{
{
IkReal j1eval[2];
IkReal x30=(cj2*j0);
IkReal x31=((0.012)*cj4);
IkReal x32=(cj3*px);
IkReal x33=((0.2732)*sj3);
IkReal x34=((54.64)*sj3);
IkReal x35=(cj2*pz);
IkReal x36=(px*sj3);
IkReal x37=((2.4)*cj4);
IkReal x38=(pz*sj2*sj4);
IkReal x39=(j0*sj2*sj4);
j1eval[0]=(((cj3*x30*x37))+(((-1.0)*x36*x37))+((x34*x35))+(((-1.0)*x30*x34))+(((-54.64)*x32))+(((-69.0)*px))+(((-1.0)*pz))+(((-1.0)*cj3*x35*x37))+(((-2.4)*x39))+(((2.4)*x38))+j0);
j1eval[1]=IKsign(((((-0.345)*px))+((cj3*x30*x31))+(((-1.0)*cj3*x31*x35))+(((-1.0)*x30*x33))+(((-0.012)*x39))+(((-1.0)*x31*x36))+(((0.005)*j0))+(((-0.005)*pz))+(((-0.2732)*x32))+(((0.012)*x38))+((x33*x35))));
if( IKabs(j1eval[0]) < 0.0000010000000000  || IKabs(j1eval[1]) < 0.0000010000000000  )
{
continue; // no branches [j1]

} else
{
{
IkReal j1array[1], cj1array[1], sj1array[1];
bool j1valid[1]={false};
_nj1 = 1;
IkReal x40=cj3*cj3;
IkReal x41=cj4*cj4;
IkReal x42=(sj2*sj4);
IkReal x43=(cj4*sj3);
IkReal x44=((0.012)*pz);
IkReal x45=(cj2*cj3);
IkReal x46=(cj2*cj4);
IkReal x47=((0.012)*j0);
IkReal x48=((0.000144)*x41);
IkReal x49=((0.2732)*cj2*sj3);
CheckValue<IkReal> x50 = IKatan2WithCheck(IkReal(((0.119025)+(((0.07463824)*x40))+(((-1.0)*x40*x48))+(((-1.0)*(j0*j0)))+(((0.188508)*cj3))+(((2.0)*j0*pz))+x48+(((0.00828)*x43))+(((-1.0)*(pz*pz)))+(((0.0065568)*cj3*x43)))),IkReal(((-0.001725)+(((-0.00414)*cj4*x45))+(((0.00414)*x42))+(((0.000144)*x42*x43))+((j0*px))+(((-1.0)*sj3*x45*x48))+(((0.0032784)*x46))+(((0.094254)*cj2*sj3))+(((0.07463824)*sj3*x45))+(((-6.0e-5)*x43))+(((-1.0)*px*pz))+(((-0.0065568)*x40*x46))+(((0.0032784)*cj3*x42))+(((-0.001366)*cj3)))),IKFAST_ATAN2_MAGTHRESH);
if(!x50.valid){
continue;
}
CheckValue<IkReal> x51=IKPowWithIntegerCheck(IKsign(((((-0.345)*px))+((pz*x49))+(((-1.0)*x42*x47))+(((-0.012)*px*x43))+(((-1.0)*j0*x49))+((cj4*x45*x47))+(((0.005)*j0))+(((-0.2732)*cj3*px))+((x42*x44))+(((-1.0)*cj4*x44*x45))+(((-0.005)*pz)))),-1);
if(!x51.valid){
continue;
}
j1array[0]=((-1.5707963267949)+(x50.value)+(((1.5707963267949)*(x51.value))));
sj1array[0]=IKsin(j1array[0]);
cj1array[0]=IKcos(j1array[0]);
if( j1array[0] > IKPI )
{
    j1array[0]-=IK2PI;
}
else if( j1array[0] < -IKPI )
{    j1array[0]+=IK2PI;
}
j1valid[0] = true;
for(int ij1 = 0; ij1 < 1; ++ij1)
{
if( !j1valid[ij1] )
{
    continue;
}
_ij1[0] = ij1; _ij1[1] = -1;
for(int iij1 = ij1+1; iij1 < 1; ++iij1)
{
if( j1valid[iij1] && IKabs(cj1array[ij1]-cj1array[iij1]) < IKFAST_SOLUTION_THRESH && IKabs(sj1array[ij1]-sj1array[iij1]) < IKFAST_SOLUTION_THRESH )
{
    j1valid[iij1]=false; _ij1[1] = iij1; break; 
}
}
j1 = j1array[ij1]; cj1 = cj1array[ij1]; sj1 = sj1array[ij1];
{
IkReal evalcond[5];
IkReal x52=IKcos(j1);
IkReal x53=IKsin(j1);
IkReal x54=(cj2*sj3);
IkReal x55=((1.0)*pz);
IkReal x56=((0.2732)*cj3);
IkReal x57=((0.012)*sj2*sj4);
IkReal x58=((0.012)*cj4*sj3);
IkReal x59=((0.2732)*x52);
IkReal x60=(pz*x52);
IkReal x61=((0.01)*x53);
IkReal x62=(px*x53);
IkReal x63=(j0*x52);
IkReal x64=(px*x52);
IkReal x65=((0.012)*cj2*cj3*cj4);
evalcond[0]=((0.345)+(((-1.0)*x52*x55))+x58+x56+x62+x63);
evalcond[1]=((0.005)+(((-1.0)*x53*x55))+((j0*x53))+x65+(((-0.2732)*x54))+(((-1.0)*x57))+(((-1.0)*x64)));
evalcond[2]=((((-1.0)*x53*x58))+(((-1.0)*x53*x56))+(((-0.345)*x53))+(((-1.0)*x52*x57))+(((0.005)*x52))+(((-1.0)*x54*x59))+(((-1.0)*px))+((x52*x65)));
evalcond[3]=((((-1.0)*x53*x57))+(((-0.2732)*x53*x54))+(((0.005)*x53))+(((0.345)*x52))+((x53*x65))+((x52*x56))+((x52*x58))+(((-1.0)*x55))+j0);
evalcond[4]=((-0.04426776)+(((0.01)*x64))+(((-1.0)*(px*px)))+(((-1.0)*j0*x61))+(((-1.0)*(j0*j0)))+(((-0.69)*x62))+(((-0.69)*x63))+(((2.0)*j0*pz))+(((-1.0)*pz*x55))+(((0.69)*x60))+((pz*x61))+(((-1.0)*(py*py))));
if( IKabs(evalcond[0]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[1]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[2]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[3]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[4]) > IKFAST_EVALCOND_THRESH  )
{
continue;
}
}

{
std::vector<IkSingleDOFSolutionBase<IkReal> > vinfos(5);
vinfos[0].jointtype = 17;
vinfos[0].foffset = j0;
vinfos[0].indices[0] = _ij0[0];
vinfos[0].indices[1] = _ij0[1];
vinfos[0].maxsolutions = _nj0;
vinfos[1].jointtype = 1;
vinfos[1].foffset = j1;
vinfos[1].indices[0] = _ij1[0];
vinfos[1].indices[1] = _ij1[1];
vinfos[1].maxsolutions = _nj1;
vinfos[2].jointtype = 1;
vinfos[2].foffset = j2;
vinfos[2].indices[0] = _ij2[0];
vinfos[2].indices[1] = _ij2[1];
vinfos[2].maxsolutions = _nj2;
vinfos[3].jointtype = 1;
vinfos[3].foffset = j3;
vinfos[3].indices[0] = _ij3[0];
vinfos[3].indices[1] = _ij3[1];
vinfos[3].maxsolutions = _nj3;
vinfos[4].jointtype = 1;
vinfos[4].foffset = j4;
vinfos[4].indices[0] = _ij4[0];
vinfos[4].indices[1] = _ij4[1];
vinfos[4].maxsolutions = _nj4;
std::vector<int> vfree(0);
solutions.AddSolution(vinfos,vfree);
}
}
}

}

}

} else
{
{
IkReal j1array[1], cj1array[1], sj1array[1];
bool j1valid[1]={false};
_nj1 = 1;
IkReal x66=cj4*cj4;
IkReal x67=cj3*cj3;
IkReal x68=(cj4*sj3);
IkReal x69=(cj2*cj3);
IkReal x70=(sj2*sj4);
IkReal x71=(cj2*cj4);
IkReal x72=((0.012)*px);
IkReal x73=(cj2*sj3);
IkReal x74=((0.2732)*cj3);
IkReal x75=((0.000144)*x66);
CheckValue<IkReal> x76=IKPowWithIntegerCheck(IKsign(((((0.012)*pz*x68))+((pz*x74))+(((0.345)*pz))+(((-0.345)*j0))+(((-0.005)*px))+((x70*x72))+(((-0.012)*j0*x68))+(((-1.0)*j0*x74))+(((-1.0)*cj4*x69*x72))+(((0.2732)*px*x73)))),-1);
if(!x76.valid){
continue;
}
CheckValue<IkReal> x77 = IKatan2WithCheck(IkReal(((0.001725)+(((-0.0032784)*x71))+(((0.00414)*cj4*x69))+(((-0.0032784)*cj3*x70))+(((-0.000144)*x68*x70))+(((-0.094254)*x73))+((j0*px))+(((0.001366)*cj3))+(((-0.07463824)*sj3*x69))+(((0.0065568)*x67*x71))+(((-0.00414)*x70))+(((6.0e-5)*x68))+((sj3*x69*x75))+(((-1.0)*px*pz)))),IkReal(((0.119025)+(((-1.0)*(px*px)))+(((0.00828)*x68))+(((0.188508)*cj3))+x75+(((0.07463824)*x67))+(((-1.0)*x67*x75))+(((0.0065568)*cj3*x68)))),IKFAST_ATAN2_MAGTHRESH);
if(!x77.valid){
continue;
}
j1array[0]=((-1.5707963267949)+(((1.5707963267949)*(x76.value)))+(x77.value));
sj1array[0]=IKsin(j1array[0]);
cj1array[0]=IKcos(j1array[0]);
if( j1array[0] > IKPI )
{
    j1array[0]-=IK2PI;
}
else if( j1array[0] < -IKPI )
{    j1array[0]+=IK2PI;
}
j1valid[0] = true;
for(int ij1 = 0; ij1 < 1; ++ij1)
{
if( !j1valid[ij1] )
{
    continue;
}
_ij1[0] = ij1; _ij1[1] = -1;
for(int iij1 = ij1+1; iij1 < 1; ++iij1)
{
if( j1valid[iij1] && IKabs(cj1array[ij1]-cj1array[iij1]) < IKFAST_SOLUTION_THRESH && IKabs(sj1array[ij1]-sj1array[iij1]) < IKFAST_SOLUTION_THRESH )
{
    j1valid[iij1]=false; _ij1[1] = iij1; break; 
}
}
j1 = j1array[ij1]; cj1 = cj1array[ij1]; sj1 = sj1array[ij1];
{
IkReal evalcond[5];
IkReal x78=IKcos(j1);
IkReal x79=IKsin(j1);
IkReal x80=(cj2*sj3);
IkReal x81=((1.0)*pz);
IkReal x82=((0.2732)*cj3);
IkReal x83=((0.012)*sj2*sj4);
IkReal x84=((0.012)*cj4*sj3);
IkReal x85=((0.2732)*x78);
IkReal x86=(pz*x78);
IkReal x87=((0.01)*x79);
IkReal x88=(px*x79);
IkReal x89=(j0*x78);
IkReal x90=(px*x78);
IkReal x91=((0.012)*cj2*cj3*cj4);
evalcond[0]=((0.345)+(((-1.0)*x78*x81))+x88+x89+x82+x84);
evalcond[1]=((0.005)+(((-0.2732)*x80))+(((-1.0)*x79*x81))+((j0*x79))+x91+(((-1.0)*x90))+(((-1.0)*x83)));
evalcond[2]=((((-1.0)*x80*x85))+((x78*x91))+(((-0.345)*x79))+(((-1.0)*x79*x84))+(((-1.0)*x79*x82))+(((-1.0)*px))+(((-1.0)*x78*x83))+(((0.005)*x78)));
evalcond[3]=(((x79*x91))+((x78*x84))+((x78*x82))+(((0.345)*x78))+(((-0.2732)*x79*x80))+(((-1.0)*x79*x83))+(((0.005)*x79))+(((-1.0)*x81))+j0);
evalcond[4]=((-0.04426776)+(((-1.0)*(px*px)))+((pz*x87))+(((-1.0)*(j0*j0)))+(((2.0)*j0*pz))+(((0.01)*x90))+(((-0.69)*x88))+(((-0.69)*x89))+(((-1.0)*(py*py)))+(((0.69)*x86))+(((-1.0)*j0*x87))+(((-1.0)*pz*x81)));
if( IKabs(evalcond[0]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[1]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[2]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[3]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[4]) > IKFAST_EVALCOND_THRESH  )
{
continue;
}
}

{
std::vector<IkSingleDOFSolutionBase<IkReal> > vinfos(5);
vinfos[0].jointtype = 17;
vinfos[0].foffset = j0;
vinfos[0].indices[0] = _ij0[0];
vinfos[0].indices[1] = _ij0[1];
vinfos[0].maxsolutions = _nj0;
vinfos[1].jointtype = 1;
vinfos[1].foffset = j1;
vinfos[1].indices[0] = _ij1[0];
vinfos[1].indices[1] = _ij1[1];
vinfos[1].maxsolutions = _nj1;
vinfos[2].jointtype = 1;
vinfos[2].foffset = j2;
vinfos[2].indices[0] = _ij2[0];
vinfos[2].indices[1] = _ij2[1];
vinfos[2].maxsolutions = _nj2;
vinfos[3].jointtype = 1;
vinfos[3].foffset = j3;
vinfos[3].indices[0] = _ij3[0];
vinfos[3].indices[1] = _ij3[1];
vinfos[3].maxsolutions = _nj3;
vinfos[4].jointtype = 1;
vinfos[4].foffset = j4;
vinfos[4].indices[0] = _ij4[0];
vinfos[4].indices[1] = _ij4[1];
vinfos[4].maxsolutions = _nj4;
std::vector<int> vfree(0);
solutions.AddSolution(vinfos,vfree);
}
}
}

}

}

} else
{
{
IkReal j1array[1], cj1array[1], sj1array[1];
bool j1valid[1]={false};
_nj1 = 1;
IkReal x92=((0.012)*pz);
IkReal x93=(sj2*sj4);
IkReal x94=((0.2732)*cj3);
IkReal x95=((0.012)*j0);
IkReal x96=((0.012)*px);
IkReal x97=(cj2*cj3*cj4);
IkReal x98=((0.2732)*cj2*sj3);
IkReal x99=((0.012)*cj4*sj3);
CheckValue<IkReal> x100 = IKatan2WithCheck(IkReal(((((-1.0)*j0*x98))+((x92*x93))+(((0.345)*px))+((cj4*sj3*x96))+((pz*x98))+(((0.005)*j0))+(((-1.0)*x92*x97))+((px*x94))+((x95*x97))+(((-0.005)*pz))+(((-1.0)*x93*x95)))),IkReal(((((-0.345)*pz))+(((-1.0)*pz*x94))+(((0.345)*j0))+((j0*x94))+((cj4*sj3*x95))+(((-1.0)*cj4*sj3*x92))+((x93*x96))+((px*x98))+(((-0.005)*px))+(((-1.0)*x96*x97)))),IKFAST_ATAN2_MAGTHRESH);
if(!x100.valid){
continue;
}
CheckValue<IkReal> x101=IKPowWithIntegerCheck(IKsign(((((-1.0)*(px*px)))+(((-1.0)*(j0*j0)))+(((2.0)*j0*pz))+(((-1.0)*(pz*pz))))),-1);
if(!x101.valid){
continue;
}
j1array[0]=((-1.5707963267949)+(x100.value)+(((1.5707963267949)*(x101.value))));
sj1array[0]=IKsin(j1array[0]);
cj1array[0]=IKcos(j1array[0]);
if( j1array[0] > IKPI )
{
    j1array[0]-=IK2PI;
}
else if( j1array[0] < -IKPI )
{    j1array[0]+=IK2PI;
}
j1valid[0] = true;
for(int ij1 = 0; ij1 < 1; ++ij1)
{
if( !j1valid[ij1] )
{
    continue;
}
_ij1[0] = ij1; _ij1[1] = -1;
for(int iij1 = ij1+1; iij1 < 1; ++iij1)
{
if( j1valid[iij1] && IKabs(cj1array[ij1]-cj1array[iij1]) < IKFAST_SOLUTION_THRESH && IKabs(sj1array[ij1]-sj1array[iij1]) < IKFAST_SOLUTION_THRESH )
{
    j1valid[iij1]=false; _ij1[1] = iij1; break; 
}
}
j1 = j1array[ij1]; cj1 = cj1array[ij1]; sj1 = sj1array[ij1];
{
IkReal evalcond[5];
IkReal x102=IKcos(j1);
IkReal x103=IKsin(j1);
IkReal x104=(cj2*sj3);
IkReal x105=((1.0)*pz);
IkReal x106=((0.2732)*cj3);
IkReal x107=((0.012)*sj2*sj4);
IkReal x108=((0.012)*cj4*sj3);
IkReal x109=((0.2732)*x102);
IkReal x110=(pz*x102);
IkReal x111=((0.01)*x103);
IkReal x112=(px*x103);
IkReal x113=(j0*x102);
IkReal x114=(px*x102);
IkReal x115=((0.012)*cj2*cj3*cj4);
evalcond[0]=((0.345)+x113+x112+x108+x106+(((-1.0)*x102*x105)));
evalcond[1]=((0.005)+(((-1.0)*x107))+x115+(((-0.2732)*x104))+(((-1.0)*x103*x105))+(((-1.0)*x114))+((j0*x103)));
evalcond[2]=((((-1.0)*x104*x109))+(((0.005)*x102))+((x102*x115))+(((-1.0)*px))+(((-1.0)*x103*x106))+(((-1.0)*x103*x108))+(((-1.0)*x102*x107))+(((-0.345)*x103)));
evalcond[3]=((((0.005)*x103))+(((-1.0)*x105))+((x103*x115))+((x102*x106))+((x102*x108))+(((-0.2732)*x103*x104))+(((0.345)*x102))+(((-1.0)*x103*x107))+j0);
evalcond[4]=((-0.04426776)+(((-1.0)*pz*x105))+(((-1.0)*(px*px)))+(((-1.0)*j0*x111))+(((-0.69)*x112))+(((-0.69)*x113))+((pz*x111))+(((-1.0)*(j0*j0)))+(((2.0)*j0*pz))+(((-1.0)*(py*py)))+(((0.01)*x114))+(((0.69)*x110)));
if( IKabs(evalcond[0]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[1]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[2]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[3]) > IKFAST_EVALCOND_THRESH  || IKabs(evalcond[4]) > IKFAST_EVALCOND_THRESH  )
{
continue;
}
}

{
std::vector<IkSingleDOFSolutionBase<IkReal> > vinfos(5);
vinfos[0].jointtype = 17;
vinfos[0].foffset = j0;
vinfos[0].indices[0] = _ij0[0];
vinfos[0].indices[1] = _ij0[1];
vinfos[0].maxsolutions = _nj0;
vinfos[1].jointtype = 1;
vinfos[1].foffset = j1;
vinfos[1].indices[0] = _ij1[0];
vinfos[1].indices[1] = _ij1[1];
vinfos[1].maxsolutions = _nj1;
vinfos[2].jointtype = 1;
vinfos[2].foffset = j2;
vinfos[2].indices[0] = _ij2[0];
vinfos[2].indices[1] = _ij2[1];
vinfos[2].maxsolutions = _nj2;
vinfos[3].jointtype = 1;
vinfos[3].foffset = j3;
vinfos[3].indices[0] = _ij3[0];
vinfos[3].indices[1] = _ij3[1];
vinfos[3].maxsolutions = _nj3;
vinfos[4].jointtype = 1;
vinfos[4].foffset = j4;
vinfos[4].indices[0] = _ij4[0];
vinfos[4].indices[1] = _ij4[1];
vinfos[4].maxsolutions = _nj4;
std::vector<int> vfree(0);
solutions.AddSolution(vinfos,vfree);
}
}
}

}

}
}
}
}
}

}

}
}
return solutions.GetNumSolutions()>0;
}
};


/// solves the inverse kinematics equations.
/// \param pfree is an array specifying the free joints of the chain.
IKFAST_API bool ComputeIk(const IkReal* eetrans, const IkReal* eerot, const IkReal* pfree, IkSolutionListBase<IkReal>& solutions) {
IKSolver solver;
return solver.ComputeIk(eetrans,eerot,pfree,solutions);
}

IKFAST_API bool ComputeIk2(const IkReal* eetrans, const IkReal* eerot, const IkReal* pfree, IkSolutionListBase<IkReal>& solutions, void* pOpenRAVEManip) {
IKSolver solver;
return solver.ComputeIk(eetrans,eerot,pfree,solutions);
}

IKFAST_API const char* GetKinematicsHash() { return "1364dab648538da052b25e6bdbf543c1"; }

IKFAST_API const char* GetIkFastVersion() { return "0x10000049"; }

#ifdef IKFAST_NAMESPACE
} // end namespace
#endif

#ifndef IKFAST_NO_MAIN
#include <stdio.h>
#include <stdlib.h>
#ifdef IKFAST_NAMESPACE
using namespace IKFAST_NAMESPACE;
#endif
int main(int argc, char** argv)
{
    if( argc != 12+GetNumFreeParameters()+1 ) {
        printf("\nUsage: ./ik r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2 free0 ...\n\n"
               "Returns the ik solutions given the transformation of the end effector specified by\n"
               "a 3x3 rotation R (rXX), and a 3x1 translation (tX).\n"
               "There are %d free parameters that have to be specified.\n\n",GetNumFreeParameters());
        return 1;
    }

    IkSolutionList<IkReal> solutions;
    std::vector<IkReal> vfree(GetNumFreeParameters());
    IkReal eerot[9],eetrans[3];
    eerot[0] = atof(argv[1]); eerot[1] = atof(argv[2]); eerot[2] = atof(argv[3]); eetrans[0] = atof(argv[4]);
    eerot[3] = atof(argv[5]); eerot[4] = atof(argv[6]); eerot[5] = atof(argv[7]); eetrans[1] = atof(argv[8]);
    eerot[6] = atof(argv[9]); eerot[7] = atof(argv[10]); eerot[8] = atof(argv[11]); eetrans[2] = atof(argv[12]);
    for(std::size_t i = 0; i < vfree.size(); ++i)
        vfree[i] = atof(argv[13+i]);
    bool bSuccess = ComputeIk(eetrans, eerot, vfree.size() > 0 ? &vfree[0] : NULL, solutions);

    if( !bSuccess ) {
        fprintf(stderr,"Failed to get ik solution\n");
        return -1;
    }

    printf("Found %d ik solutions:\n", (int)solutions.GetNumSolutions());
    std::vector<IkReal> solvalues(GetNumJoints());
    for(std::size_t i = 0; i < solutions.GetNumSolutions(); ++i) {
        const IkSolutionBase<IkReal>& sol = solutions.GetSolution(i);
        printf("sol%d (free=%d): ", (int)i, (int)sol.GetFree().size());
        std::vector<IkReal> vsolfree(sol.GetFree().size());
        sol.GetSolution(&solvalues[0],vsolfree.size()>0?&vsolfree[0]:NULL);
        for( std::size_t j = 0; j < solvalues.size(); ++j)
            printf("%.15f, ", solvalues[j]);
        printf("\n");
    }
    return 0;
}

#endif