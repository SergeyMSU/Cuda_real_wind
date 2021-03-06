
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include <cmath>

#define L 1.0
#define minut 0.000150403  // 1 ?????? ? ???????????? ???????.
#define hour 0.00902419      // 1 ??? ? ???????????? ???????.
#define day 0.216581      // 1 ???? ? ???????????? ???????.
#define dx 0.00125  //0.00025  // 0.0001         // ?????? ?????? ? ?.?.
#define THREADS_PER_BLOCK 256
#define Omni_ 20000 //9650


#define ga (5.0/3.0)          // ?????????? ????????
#define ggg (5.0/3.0)
#define kv(x) ((x)*(x))
#define kvv(x,y,z)  (kv(x) + kv(y) + kv(z))
#define U8(ro, p, u, v, w, bx, by, bz)  ( (p) / (ggg - 1.0) + 0.5 * (ro) * kvv((u),(v),(w)) + kvv((bx),(by),(bz)) / cpi8)
#define skk(u,v,w,bx,by,bz) (u*bx + v*by + w*bz)
#define g1 (ga - 1.0)
#define gg1 (ga - 1.0)
#define g2 (ga + 1.0)
#define gg2 (ga + 1.0)
#define gp ((g2/ga)/2.0)
#define gm ((g1/ga)/2.0)
#define gga ga
#define Omega 0.0
#define eps 10e-10
#define eps8 10e-8
#define kurant  0.2 //0.2
#define pi 3.14159265358979323846
#define PI 3.14159265358979323846
#define cpi4 12.56637061435917295384
#define cpi8 25.13274122871834590768
#define spi4 __dsqrt_rn(cpi4)
#define epsb 1e-6
#define eps_p 1e-6
#define eps_d 1e-3
#define krit 0.2  // 0.2



#define a_2 0.162294  // 0.10263
#define sigma(x) (kv(1.0 - a_2 * log(x)))
#define br(x) (0.0/( (x) * (x) )) // (0.155895/( (x) * (x) ))

#define c_H 0.0391451
#define v_H -0.0704
#define Kn 0.392914    // ???????????? ??????????? ????? ??????????? ???????
#define n_H 0.00714286   // 0.05  ???????????? ???????????? ?????? ????????
#define r0 2.7268   // ???????????? r0 ? ????????? ???????? ? ??????????

#define data_voy 127391  // 127476  // ?????????? ????????? ????????

using namespace std;

__device__ double linear(const double& x1, const double& t1, const double& x2, const double& t2, const double& y);
__device__ double HLLC_2d_Korolkov_b_s(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L,//
    const double& pp_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& pp_R, const double& W, //
    double* P, double& PQ, const double& n1, const double& n2, const double& rad, double& RO_p, double& P_p, int metod = 1, bool nul_potok = false);
__device__ double minmod(const double& x, const double& y);
__device__ double sign(const double& x);

cudaError_t addWithCuda(double* ro, double* p, double* u, int& N);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void funk_time(double* T, double* T_do, double* TT)
{
    *TT = *TT + *T_do;
    *T_do = *T;
    *T = 0.1 * day;// 0.5 * minut;
    return;
}


__global__ void funk_time_devide(double* T_do)
{
    *T_do = *T_do / 4.0;
    return;
}

__device__ double HLLC_2d_Korolkov_b_s(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L,//
    const double& pp_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& pp_R, const double& W, //
    double* P, double& PQ, const double& n1, const double& n2, const double& rad, double& RO_p, double& P_p, int metod, bool nul_potok)
    // BestSeries
    // ?????? ?????????? 2? ?????????
    //
    //  ?????:
    // P[1]       // ????????
    // P[2]
    // P[0]       // ?????
    // P[3]       // ???????
{
    double t1 = n2;
    double t2 = -n1;

    double rop_L = ro_L;// -Q_L;
    double rop_R = ro_R;// -Q_R;

    double u1, v1, u2, v2;
    u1 = v1_L * n1 + v2_L * n2;
    v1 = v1_L * t1 + v2_L * t2;
    u2 = v1_R * n1 + v2_R * n2;
    v2 = v1_R * t1 + v2_R * t2;

    double sqrtroL = sqrt(ro_L);
    double sqrtroR = sqrt(ro_R);
    double cL = sqrt(ggg * p_L / ro_L);
    double cR = sqrt(ggg * p_R / ro_R);


    double uu_L = (kv(v1_L) + kv(v2_L)) / 2.0;
    double uu_R = (kv(v1_R) + kv(v2_R)) / 2.0;



    double SL = min(u1, u2) - max(cL, cR);
    double SR = max(u1, u2) + max(cL, cR);
    double suR = (SR - u2);
    double suL = (SL - u1);

    double SM = (suR * ro_R * u2 - suL * ro_L * u1 - p_R + p_L) //
        / (suR * ro_R - suL * ro_L);

    //double PTT = (suR * ro_R * p_L - suL * ro_L * p_R + ro_L * ro_R * suR * suL * (u2 - u1)) / (suR * ro_R - suL * ro_L);

    double UU = max(fabs(SL), fabs(SR));
    double time = kurant * rad / UU;

    double FL[6], FR[6], UL[6], UR[6];

    double e1 = p_L / g1 + ro_L * uu_L;
    double e2 = p_R / g1 + ro_R * uu_R;
    double ep1 = pp_L / g1 + rop_L * uu_L;
    double ep2 = pp_R / g1 + rop_R * uu_R;


    FL[0] = ro_L * u1;
    FL[1] = ro_L * u1 * u1 + p_L;
    FL[2] = ro_L * u1 * v1;
    FL[3] = (e1 + p_L) * u1;
    FL[4] = Q_L * u1;
    FL[5] = (ep1 + pp_L) * u1;

    FR[0] = ro_R * u2;
    FR[1] = ro_R * u2 * u2 + p_R;
    FR[2] = ro_R * u2 * v2;
    FR[3] = (e2 + p_R) * u2;
    FR[4] = Q_R * u2;
    FR[5] = (ep2 + pp_R) * u2;


    UL[0] = ro_L;
    UL[1] = ro_L * u1;
    UL[2] = ro_L * v1;
    UL[3] = e1;
    UL[4] = Q_L;
    UL[5] = ep1;

    UR[0] = ro_R;
    UR[1] = ro_R * u2;
    UR[2] = ro_R * v2;
    UR[3] = e2;
    UR[4] = Q_R;
    UR[5] = ep2;

    if (SL >= W)
    {
        P[1] = n1 * (FL[1] - W * UL[1]) + t1 * (FL[2] - W * UL[2]);     // ????????
        P[2] = n2 * (FL[1] - W * UL[1]) + t2 * (FL[2] - W * UL[2]);
        P[0] = FL[0] - W * UL[0];                       // ?????
        P[3] = FL[3] - W * UL[3];                       // ???????
        PQ = FL[4] - W * UL[4];
        P[4] = FL[5] - W * UL[5];                       // ??? ??????? ????????
        RO_p = rop_L;
        P_p = pp_L;
        return time;
    }

    if (SR <= W)
    {
        P[1] = n1 * (FR[1] - W * UR[1]) + t1 * (FR[2] - W * UR[2]);     // ????????
        P[2] = n2 * (FR[1] - W * UR[1]) + t2 * (FR[2] - W * UR[2]);
        P[0] = FR[0] - W * UR[0];                       // ?????
        P[3] = FR[3] - W * UR[3];                       // ???????
        PQ = FR[4] - W * UR[4];
        P[4] = FR[5] - W * UR[5];
        RO_p = rop_R;
        P_p = pp_R;
        return time;
    }

    //printf("TUT\n");
    double ro_LL = ro_L * (SL - u1) / (SL - SM);
    double ro_RR = ro_R * (SR - u2) / (SR - SM);
    double Q_LL = Q_L * (SL - u1) / (SL - SM);
    double Q_RR = Q_R * (SR - u2) / (SR - SM);
    double rop_LL = rop_L * (SL - u1) / (SL - SM);
    double rop_RR = rop_R * (SR - u2) / (SR - SM);


    double UZ0 = (SR * UR[0] - SL * UL[0] + FL[0] - FR[0]) / (SR - SL);
    double UZ1 = (SR * UR[1] - SL * UL[1] + FL[1] - FR[1]) / (SR - SL);
    double UZ2 = (SR * UR[2] - SL * UL[2] + FL[2] - FR[2]) / (SR - SL);
    double UZ3 = (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) / (SR - SL);
    double UZ4 = (SR * UR[4] - SL * UL[4] + FL[4] - FR[4]) / (SR - SL);
    double UZ5 = (SR * UR[5] - SL * UL[5] + FL[5] - FR[5]) / (SR - SL);
    double vzL, vzR, vLL, vRR, ppLR, ee1, ee2, eep1, eep2;

    // ??? ?????????? ?? ???????? ??? ?????? ? ???????????? ????????
    //if (metod == 0)
    //{
    //    double  PO[5];
    //    for (int i = 0; i < 6; i++)
    //    {
    //        PO[i] = (SR * FL[i] - SL * FR[i] + SR * SL * (UR[i] - UL[i])) / (SR - SL);
    //    }

    //    P[1] = n1 * (PO[1] - W * UZ1) + t1 * (PO[2] - W * UZ2);     // ????????
    //    P[2] = n2 * (PO[1] - W * UZ1) + t2 * (PO[2] - W * UZ2);
    //    P[0] = PO[0] - W * UZ0;                       // ?????
    //    P[3] = PO[3] - W * UZ3;                       // ???????
    //    PQ = PO[4] - W * UZ4;
    //    P[4] = PO[5] - W * UZ5;
    //    return time;
    //}


    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = ro_R * suRm;
    double rzL = ro_L * suLm;

    double ptzR = p_R + ro_R * suR * (SM - u2);
    double ptzL = p_L + ro_L * suL * (SM - u1);
    double ptz = (ptzR + ptzL) / 2.0;

    double ptzpR = pp_R + rop_R * suR * (SM - u2);
    double ptzpL = pp_L + rop_L * suL * (SM - u1);
    double ptzp = (ptzpR + ptzpL) / 2.0;

    P_p = ptzp;
    /*if( fabs(v1 - v2) > 0.1)
    {
        vLL = v1;
        vRR = v2;
    }
    else
    {
        vRR = UZ2 / UZ0;
        vLL = vRR;
    }*/


    if (nul_potok == true)   // ????? ???????????
    {
        vRR = UZ2 / UZ0;
        vLL = vRR;
    }
    else
    {
        vLL = v1;
        vRR = v2;
    }



    ee2 = e2 * suRm + (ptz * SM - p_R * u2) / (SR - SM);
    ee1 = e1 * suLm + (ptz * SM - p_L * u1) / (SL - SM);
    eep2 = ep2 * suRm + (ptzp * SM - pp_R * u2) / (SR - SM);
    eep1 = ep1 * suLm + (ptzp * SM - pp_L * u1) / (SL - SM);


    double  ULL[6], URR[6], PO[6];
    ULL[0] = ro_LL;
    ULL[1] = ro_LL * SM;
    ULL[2] = ro_LL * vLL;
    ULL[3] = ee1;
    ULL[4] = Q_LL;
    ULL[5] = eep1;

    URR[0] = ro_RR;
    URR[1] = ro_RR * SM;
    URR[2] = ro_RR * vRR;
    URR[3] = ee2;
    URR[4] = Q_RR;
    URR[5] = eep2;

    if (SL < W && SM >= W)
    {
        for (int i = 0; i < 6; i++)
        {
            PO[i] = FL[i] + SL * ULL[i] - SL * UL[i] - W * ULL[i];
        }
        RO_p = rop_LL;
    }
    else if (SR > W && SM < W)
    {
        for (int i = 0; i < 6; i++)
        {
            PO[i] = FR[i] + SR * URR[i] - SR * UR[i] - W * URR[i];
        }
        RO_p = rop_RR;
    }

    P[1] = n1 * PO[1] + t1 * PO[2];     // ????????
    P[2] = n2 * PO[1] + t2 * PO[2];
    P[0] = PO[0];                       // ?????
    P[3] = PO[3];                       // ???????
    PQ = PO[4];
    P[4] = PO[5];
    return time;
}

__device__ double HLLDQ_Korolkov(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, double& PQ, const double& n1, const double& n2, const double& n3, const double& rad, int metod)
{// ?? ????????, ???? ???????? ????? ?? ???????
 // ??????? ????? ????????? ?? ???? ????????? ! (????? ????? ??????? ???????????? ??????????? ????????)

    double bx_L = Bx_L / spi4;
    double by_L = By_L / spi4;
    double bz_L = Bz_L / spi4;

    double bx_R = Bx_R / spi4;
    double by_R = By_R / spi4;
    double bz_R = Bz_R / spi4;

    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    double m1 = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;

    if (n1 > 0.1)
    {
        t2 = 1.0;
        m3 = 1.0;
    }
    else if (n2 > 0.1)
    {
        t3 = 1.0;
        m1 = 1.0;
    }
    else if (n3 > 0.1)
    {
        t1 = 1.0;
        m2 = 1.0;
    }
    else if (n1 < -0.1)
    {
        t3 = -1.0;
        m2 = -1.0;
    }
    else if (n2 < -0.1)
    {
        t1 = -1.0;
        m3 = -1.0;
    }
    else if (n3 < -0.1)
    {
        t1 = -1.0;
        m2 = -1.0;
    }
    else
    {
        printf("EROROR 1421  normal_error\n");
    }


    double u1, v1, w1, u2, v2, w2;
    u1 = v1_L * n1 + v2_L * n2 + v3_L * n3;
    v1 = v1_L * t1 + v2_L * t2 + v3_L * t3;
    w1 = v1_L * m1 + v2_L * m2 + v3_L * m3;
    u2 = v1_R * n1 + v2_R * n2 + v3_R * n3;
    v2 = v1_R * t1 + v2_R * t2 + v3_R * t3;
    w2 = v1_R * m1 + v2_R * m2 + v3_R * m3;

    double bn1, bt1, bm1, bn2, bt2, bm2;
    bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
    bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
    bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
    bn2 = bx_R * n1 + by_R * n2 + bz_R * n3;
    bt2 = bx_R * t1 + by_R * t2 + bz_R * t3;
    bm2 = bx_R * m1 + by_R * m2 + bz_R * m3;

    //cout << " = " << bt2 * bt2 + bm2 * bm2 << endl;

    double sqrtroL = sqrt(ro_L);
    double sqrtroR = sqrt(ro_R);
    double ca_L = bn1 / sqrtroL;
    double ca_R = bn2 / sqrtroR;
    double cL = sqrt(ggg * p_L / ro_L);
    double cR = sqrt(ggg * p_R / ro_R);

    double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
    double bb_R = kv(bx_R) + kv(by_R) + kv(bz_R);

    double aL = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;
    double aR = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;

    double uu_L = (kv(v1_L) + kv(v2_L) + kv(v3_L)) / 2.0;
    double uu_R = (kv(v1_R) + kv(v2_R) + kv(v3_R)) / 2.0;

    double cfL = sqrt((ggg * p_L + bb_L + //
        sqrt(kv(ggg * p_L + bb_L) - 4.0 * ggg * p_L * kv(bn1))) / (2.0 * ro_L));
    double cfR = sqrt((ggg * p_R + bb_R + //
        sqrt(kv(ggg * p_R + bb_R) - 4.0 * ggg * p_R * kv(bn2))) / (2.0 * ro_R));


    double SL = min(u1, u2) - max(cfL, cfR);
    double SR = max(u1, u2) + max(cfL, cfR);

    double pTL = p_L + bb_L / 2.0;
    double pTR = p_R + bb_R / 2.0;

    double suR = (SR - u2);
    double suL = (SL - u1);

    double SM = (suR * ro_R * u2 - suL * ro_L * u1 - pTR + pTL) //
        / (suR * ro_R - suL * ro_L);

    double PTT = (suR * ro_R * pTL - suL * ro_L * pTR + ro_L * ro_R * suR * suL * (u2 - u1))//
        / (suR * ro_R - suL * ro_L);

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double FL[9], FR[9], UL[9], UR[9];

    double e1 = p_L / g1 + ro_L * uu_L + bb_L / 2.0;
    double e2 = p_R / g1 + ro_R * uu_R + bb_R / 2.0;


    FL[0] = ro_L * u1;
    FL[1] = ro_L * u1 * u1 + pTL - kv(bn1);
    FL[2] = ro_L * u1 * v1 - bn1 * bt1;
    FL[3] = ro_L * u1 * w1 - bn1 * bm1;
    FL[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
    //cout << uu_L << endl;
    FL[5] = 0.0;
    FL[6] = u1 * bt1 - v1 * bn1;
    FL[7] = u1 * bm1 - w1 * bn1;
    FL[8] = Q_L * u1;

    FR[0] = ro_R * u2;
    FR[1] = ro_R * u2 * u2 + pTR - kv(bn2);
    FR[2] = ro_R * u2 * v2 - bn2 * bt2;
    FR[3] = ro_R * u2 * w2 - bn2 * bm2;
    FR[4] = (e2 + pTR) * u2 - bn2 * (u2 * bn2 + v2 * bt2 + w2 * bm2);
    FR[5] = 0.0;
    FR[6] = u2 * bt2 - v2 * bn2;
    FR[7] = u2 * bm2 - w2 * bn2;
    FR[8] = Q_R * u2;

    UL[0] = ro_L;
    UL[1] = ro_L * u1;
    UL[2] = ro_L * v1;
    UL[3] = ro_L * w1;
    UL[4] = e1;
    UL[5] = bn1;
    UL[6] = bt1;
    UL[7] = bm1;
    UL[8] = Q_L;

    UR[0] = ro_R;
    UR[1] = ro_R * u2;
    UR[2] = ro_R * v2;
    UR[3] = ro_R * w2;
    UR[4] = e2;
    UR[5] = bn2;
    UR[6] = bt2;
    UR[7] = bm2;
    UR[8] = Q_R;

    double bn = (SR * UR[5] - SL * UL[5] + FL[5] - FR[5]) / (SR - SL);
    double bt = (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) / (SR - SL);
    double bm = (SR * UR[7] - SL * UL[7] + FL[7] - FR[7]) / (SR - SL);
    double bbn = bn * bn;

    double ro_LL = ro_L * (SL - u1) / (SL - SM);
    double ro_RR = ro_R * (SR - u2) / (SR - SM);
    double Q_LL = Q_L * (SL - u1) / (SL - SM);
    double Q_RR = Q_R * (SR - u2) / (SR - SM);

    if (metod == 2)   // HLLC  + mgd
    {
        double sbv1 = u1 * bn1 + v1 * bt1 + w1 * bm1;
        double sbv2 = u2 * bn2 + v2 * bt2 + w2 * bm2;

        double UZ0 = (SR * UR[0] - SL * UL[0] + FL[0] - FR[0]) / (SR - SL);
        double UZ1 = (SR * UR[1] - SL * UL[1] + FL[1] - FR[1]) / (SR - SL);
        double UZ2 = (SR * UR[2] - SL * UL[2] + FL[2] - FR[2]) / (SR - SL);
        double UZ3 = (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) / (SR - SL);
        double UZ4 = (SR * UR[4] - SL * UL[4] + FL[4] - FR[4]) / (SR - SL);
        double vzL, vzR, vLL, wLL, vRR, wRR, ppLR, btt1, bmm1, btt2, bmm2, ee1, ee2;


        double suRm = suR / (SR - SM);
        double suLm = suL / (SL - SM);
        double rzR = ro_R * suRm;
        double rzL = ro_L * suLm;

        double ptzR = pTR + ro_R * suR * (SM - u2);
        double ptzL = pTL + ro_L * suL * (SM - u1);
        double ptz = (ptzR + ptzL) / 2.0;


        vRR = UZ2 / UZ0;
        wRR = UZ3 / UZ0;
        vLL = vRR;
        wLL = wRR;

        /*vRR = v2 + bn * (bt2 - bt) / suR / ro_R;
        wRR = w2 + bn * (bm2 - bm) / suR / ro_R;
        vLL = v1 + bn * (bt1 - bt) / suL / ro_L;
        wLL = w1 + bn * (bm1 - bm) / suL / ro_L;*/

        btt2 = bt;
        bmm2 = bm;
        btt1 = btt2;
        bmm1 = bmm2;

        double sbvz = (bn * UZ1 + bt * UZ2 + bm * UZ3) / UZ0;

        ee2 = e2 * suRm + (ptz * SM - pTR * u2 + bn * (sbv2 - sbvz)) / (SR - SM);
        ee1 = e1 * suLm + (ptz * SM - pTL * u1 + bn * (sbv1 - sbvz)) / (SL - SM);

        /*if (fabs(bn) < 0.000001 )
        {
            vRR = v2;
            wRR = w2;
            vLL = v1;
            wLL = w1;
            btt2 = bt2 * suRm;
            bmm2 = bm2 * suRm;
            btt1 = bt1 * suLm;
            bmm1 = bm1 * suLm;
        }*/

        /*ppLR = (pTL + ro_L * (SL - u1) * (SM - u1) + pTR + ro_R * (SR - u2) * (SM - u2)) / 2.0;

        if (fabs(bn) < 0.000001)
        {
            vLL = v1;
            wLL = w1;
            vRR = v2;
            wRR = w2;

            btt1 = bt1 * (SL - u1) / (SL - SM);
            btt2 = bt2 * (SR - u2) / (SR - SM);

            bmm1 = bm1 * (SL - u1) / (SL - SM);
            bmm2 = bm2 * (SR - u2) / (SR - SM);

            ee1 = ((SL - u1) * e1 - pTL * u1 + ppLR * SM) / (SL - SM);
            ee2 = ((SR - u2) * e2 - pTL * u2 + ppLR * SM) / (SR - SM);
        }
        else
        {
            btt2 = btt1 = (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) / (SR - SL);
            bmm2 = bmm1 = (SR * UR[7] - SL * UL[7] + FL[7] - FR[7]) / (SR - SL);
            vLL = v1 + bn * (bt1 - btt1) / (ro_L * (SL - u1));
            vRR = v2 + bn * (bt2 - btt2) / (ro_R * (SR - u2));

            wLL = w1 + bn * (bm1 - bmm1) / (ro_L * (SL - u1));
            wRR = w2 + bn * (bm2 - bmm2) / (ro_R * (SR - u2));

            double sks1 = u1 * bn1 + v1 * bt1 + w1 * bm1 - SM * bn - vLL * btt1 - wLL * bmm1;
            double sks2 = u2 * bn2 + v2 * bt2 + w2 * bm2 - SM * bn - vRR * btt2 - wRR * bmm2;

            ee1 = ((SL - u1) * e1 - pTL * u1 + ppLR * SM + bn * sks1) / (SL - SM);
            ee2 = ((SR - u2) * e2 - pTR * u2 + ppLR * SM + bn * sks2) / (SR - SM);
        }*/


        double  ULL[9], URR[9], PO[9];
        ULL[0] = ro_LL;
        ULL[1] = ro_LL * SM;
        ULL[2] = ro_LL * vLL;
        ULL[3] = ro_LL * wLL;
        ULL[4] = ee1;
        ULL[5] = bn;
        ULL[6] = btt1;
        ULL[7] = bmm1;
        ULL[8] = Q_LL;

        URR[0] = ro_RR;
        URR[1] = ro_RR * SM;
        URR[2] = ro_RR * vRR;
        URR[3] = ro_RR * wRR;
        URR[4] = ee2;
        URR[5] = bn;
        URR[6] = btt2;
        URR[7] = bmm2;
        URR[8] = Q_RR;

        if (SL >= 0.0)
        {
            /*if (ro_L * kv(u1 - SL) / kv(bn1) <= 1.0 || ro_R * kv(u2 - SL) / kv(bn2) <= 1.0)
            {
                printf("%lf    __    %lf\n", ro_L * kv(u1 - SL) / kv(bn1), ro_R * kv(u2 - SL) / kv(bn2));
            }*/

            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i];
            }
        }
        else if (SL < 0.0 && SM >= 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
            }
        }
        else if (SR > 0.0 && SM < 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
            }
        }
        else if (SR <= 0.0)
        {
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i];
            }
        }



        double SN = max(fabs(SL), fabs(SR));

        PO[5] = -SN * (bn2 - bn1);

        P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
        P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
        P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
        P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
        P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
        P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
        P[0] = PO[0];
        P[4] = PO[4];
        PQ = PO[8];

        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;
        return time;

    }
    else if (metod == 3)  // HLLD
    {

        double ttL = ro_L * suL * (SL - SM) - bbn;
        double ttR = ro_R * suR * (SR - SM) - bbn;

        double vLL, wLL, vRR, wRR, btt1, bmm1, btt2, bmm2;

        if (fabs(ttL) >= 0.000001)
        {
            vLL = v1 - bn * bt1 * (SM - u1) / ttL;
            wLL = w1 - bn * bm1 * (SM - u1) / ttL;
            btt1 = bt1 * (ro_L * suL * suL - bbn) / ttL;
            bmm1 = bm1 * (ro_L * suL * suL - bbn) / ttL;
        }
        else
        {
            vLL = v1;
            wLL = w1;
            btt1 = 0.0;
            bmm1 = 0.0;
        }

        if (fabs(ttR) >= 0.000001)
        {
            vRR = v2 - bn * bt2 * (SM - u2) / ttR;
            wRR = w2 - bn * bm2 * (SM - u2) / ttR;
            btt2 = bt2 * (ro_R * suR * suR - bbn) / ttR;
            bmm2 = bm2 * (ro_R * suR * suR - bbn) / ttR;
            //cout << "tbr = " << (ro_R * suR * suR - bbn) / ttR << endl;
            //cout << "bt2 = " << bt2 << endl;
        }
        else
        {
            vRR = v2;
            wRR = w2;
            btt2 = 0.0;
            bmm2 = 0.0;
        }

        double eLL = (e1 * suL + PTT * SM - pTL * u1 + bn * //
            ((u1 * bn1 + v1 * bt1 + w1 * bm1) - (SM * bn + vLL * btt1 + wLL * bmm1))) //
            / (SL - SM);
        double eRR = (e2 * suR + PTT * SM - pTR * u2 + bn * //
            ((u2 * bn2 + v2 * bt2 + w2 * bm2) - (SM * bn + vRR * btt2 + wRR * bmm2))) //
            / (SR - SM);

        double sqrtroLL = sqrt(ro_LL);
        double sqrtroRR = sqrt(ro_RR);
        double SLL = SM - fabs(bn) / sqrtroLL;
        double SRR = SM + fabs(bn) / sqrtroRR;

        double idbn = 1.0;
        if (fabs(bn) > 0.0001)
        {
            idbn = 1.0 * sign(bn);
        }
        else
        {
            idbn = 0.0;
            SLL = SM;
            SRR = SM;
        }

        double vLLL = (sqrtroLL * vLL + sqrtroRR * vRR + //
            idbn * (btt2 - btt1)) / (sqrtroLL + sqrtroRR);

        double wLLL = (sqrtroLL * wLL + sqrtroRR * wRR + //
            idbn * (bmm2 - bmm1)) / (sqrtroLL + sqrtroRR);

        double bttt = (sqrtroLL * btt2 + sqrtroRR * btt1 + //
            idbn * sqrtroLL * sqrtroRR * (vRR - vLL)) / (sqrtroLL + sqrtroRR);

        double bmmm = (sqrtroLL * bmm2 + sqrtroRR * bmm1 + //
            idbn * sqrtroLL * sqrtroRR * (wRR - wLL)) / (sqrtroLL + sqrtroRR);

        double eLLL = eLL - idbn * sqrtroLL * ((SM * bn + vLL * btt1 + wLL * bmm1) //
            - (SM * bn + vLLL * bttt + wLLL * bmmm));
        double eRRR = eRR + idbn * sqrtroRR * ((SM * bn + vRR * btt2 + wRR * bmm2) //
            - (SM * bn + vLLL * bttt + wLLL * bmmm));
        //cout << " = " << bn << " " << btt2 << " " << bmm2 << endl;
        //cout << "sbvr = " << (SM * bn + vRR * btt2 + wRR * bmm2) << endl;
        double  ULL[9], URR[9], ULLL[9], URRR[9];

        ULL[0] = ro_LL;
        ULL[1] = ro_LL * SM;
        ULL[2] = ro_LL * vLL;
        ULL[3] = ro_LL * wLL;
        ULL[4] = eLL;
        ULL[5] = bn;
        ULL[6] = btt1;
        ULL[7] = bmm1;
        ULL[8] = Q_LL;

        URR[0] = ro_RR;
        //cout << ro_RR << endl;
        URR[1] = ro_RR * SM;
        URR[2] = ro_RR * vRR;
        URR[3] = ro_RR * wRR;
        URR[4] = eRR;
        URR[5] = bn;
        URR[6] = btt2;
        URR[7] = bmm2;
        URR[8] = Q_RR;

        ULLL[0] = ro_LL;
        ULLL[1] = ro_LL * SM;
        ULLL[2] = ro_LL * vLLL;
        ULLL[3] = ro_LL * wLLL;
        ULLL[4] = eLLL;
        ULLL[5] = bn;
        ULLL[6] = bttt;
        ULLL[7] = bmmm;
        ULLL[8] = Q_LL;

        URRR[0] = ro_RR;
        URRR[1] = ro_RR * SM;
        URRR[2] = ro_RR * vLLL;
        URRR[3] = ro_RR * wLLL;
        URRR[4] = eRRR;
        URRR[5] = bn;
        URRR[6] = bttt;
        URRR[7] = bmmm;
        URRR[8] = Q_RR;

        double PO[9];

        if (SL >= 0.0)
        {
            //cout << "SL >= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i];
            }
        }
        else if (SL < 0.0 && SLL >= 0.0)
        {
            //cout << "SL < 0.0 && SLL >= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
            }
            //cout << ULL[0] << endl;
        }
        else if (SLL <= 0.0 && SM >= 0.0)
        {
            //cout << "SLL <= 0.0 && SM >= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FL[i] + SLL * ULLL[i] - (SLL - SL) * ULL[i] - SL * UL[i];
            }
        }
        else if (SM < 0.0 && SRR > 0.0)
        {
            //cout << "SM < 0.0 && SRR > 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i] + SRR * URRR[i] - (SRR - SR) * URR[i] - SR * UR[i];
            }
            //cout << "P4 = " << URRR[4] << endl;
        }
        else if (SR > 0.0 && SRR <= 0.0)
        {
            //cout << "SR > 0.0 && SRR <= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
            }
            //cout << URR[0] << endl;
        }
        else if (SR <= 0.0)
        {
            //cout << "SR <= 0.0" << endl;
            for (int i = 0; i < 9; i++)
            {
                PO[i] = FR[i];
            }
        }



        double SN = max(fabs(SL), fabs(SR));

        PO[5] = -SN * (bn2 - bn1);

        P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
        P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
        P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
        P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
        P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
        P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
        P[0] = PO[0];
        P[4] = PO[4];
        PQ = PO[8];

        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;
        return time;
    }

}

__device__ double linear(const double& x1, const double& t1, const double& x2, const double& t2, const double& y)
{
    double d = (t1 - t2) / (x1 - x2);
    return  (d * (y - x2) + t2);
}

__device__ double minmod(const double& x, const double& y)
{
    if (sign(x) + sign(y) == 0)
    {
        return 0.0;
    }
    else
    {
        return   ((sign(x) + sign(y)) / 2.0) * min(fabs(x), fabs(y));  ///minmod
        //return (2*x*y)/(x + y);   /// vanleer
    }
}

__device__ double sign(const double& x)
{
    if (x > 0)
    {
        return 1.0;
    }
    else if (x < 0)
    {
        return -1.0;
    }
    else
    {
        return 0.0;
    }
}

__device__ double linear(const double& x1, const double& t1, const double& x2, const double& t2, const double& x3, const double& t3, const double& y)
// ??????? ???????? ? ??????????? 2
// ?????? ????? ????? 1 ? 2,  2 ? 3, ????? ??????? ???????? ???????? ? y
{
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    return  (d * (y - x2) + t2);
}

void takeDataOmni(double* ro, double* p, double* u, double* t, int n1, int n2)
{
    ifstream fout1;
    fout1.open("omni_all_data_day.txt"); // omni_all_data.txt   omni_all_data_hour.txt     omni_all_data_day.txt
    int a1, a2, a3, a4, a5;
    double b1, b2, b3, b4;
    string s1, s2, s3, s4;
    fout1 >> s1 >> s2 >> s3 >> s4;
    cout << s1 << endl;
    for (int i = 0; i < n2; i++)
    {
        fout1 >> a1 >> b1 >> b2 >> b3;
    }

    for (int i = 0; i < n1; i++)
    {
        fout1 >> a1 >> b1 >> b2 >> b3;
        t[i] = a1 * minut;
        u[i] = b1 / 375.0;
        ro[i] = b2 / 7.0;
        p[i] = 2.0 * ro[i] * b3 * 5.86922 * 0.00000001;
    }
    fout1.close();
}

void takeDataOmni2(double* ro, double* p, double* u, double* v, double* w, double* t, int n1, int n2)
{
    // ????? ????????? ???? ?? ???????????, ??? ?????????? ?????? ????????
    ifstream fout1;
    fout1.open("omni_all_data_hour_mgd.txt"); // omni_all_data.txt   omni_all_data_hour.txt     omni_all_data_day.txt
    int a1, a2, a3, a4, a5;
    double b1, b2, b3, b4, b5, b6, b7, b8, b9;
    string s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
    fout1 >> s1 >> s2 >> s3 >> s4 >> s5 >> s6 >> s7 >> s8 >> s9 >> s10;
    cout << s1 << endl;
    for (int i = 0; i < n2; i++)
    {
        fout1 >> a1 >> b1 >> b2 >> b3 >> b4 >> b5 >> b6 >> b7 >> b8 >> b9;
    }

    for (int i = 0; i < n1; i++)
    {
        fout1 >> a1 >> b1 >> b2 >> b3 >> b4 >> b5 >> b6 >> b7 >> b8 >> b9;
        t[i] = a1 * minut;
        u[i] = b7 / 375.0;
        v[i] = b8 / 375.0;
        w[i] = b9 / 375.0;
        ro[i] = b5 / 7.0;
        p[i] = 2.0 * ro[i] * b4 * 5.86922 * 0.00000001;
    }
    fout1.close();
}

__global__ void takeOmni(double* ro, double* p, double* ro_p, double* p_p, double* u, double* RO, double* P, double* U, double* T, double* t_now, int* dev_mas_Omni)
{
    //printf("Omni 2,   %lf,   %lf,   %d \n", T[0], *t_now, *dev_mas_Omni);
    for (int k = *dev_mas_Omni; k < Omni_; k++)
    {
        if (T[k] >= *t_now && k >= 1)
        {
            *dev_mas_Omni = k;
            ro[0] = linear(T[k - 1], RO[k - 1], T[k], RO[k], *t_now);
            u[0] = linear(T[k - 1], U[k - 1], T[k], U[k], *t_now);
            p[0] = linear(T[k - 1], P[k - 1], T[k], P[k], *t_now);
            ro_p[0] = ro[0]; // 0.000001;
            p_p[0] = p[0];
            //printf("Omni,   %lf,   %lf,   %lf,  %d \n", T[k], *t_now, T[k - 1], *dev_mas_Omni);
            break;
        }
    }

}

__global__ void takeOmni2(double* ro, double* p, double* ro_p, double* p_p, double* u, double* v, double* w,//
    double* RO, double* P, double* U, double* V, double* W, double* T, double* t_now, int* dev_mas_Omni)
{
    // ????????????? ??????? ??? ? ??????? ?????????? ????????
    //printf("Omni 2,   %lf,   %lf,   %d \n", T[0], *t_now, *dev_mas_Omni);
    for (int k = *dev_mas_Omni; k < Omni_; k++)
    {
        if (T[k] >= *t_now && k >= 1)
        {
            *dev_mas_Omni = k;
            ro[0] = linear(T[k - 1], RO[k - 1], T[k], RO[k], *t_now);
            u[0] = linear(T[k - 1], U[k - 1], T[k], U[k], *t_now);
            v[0] = linear(T[k - 1], V[k - 1], T[k], V[k], *t_now);
            w[0] = linear(T[k - 1], W[k - 1], T[k], W[k], *t_now);
            p[0] = linear(T[k - 1], P[k - 1], T[k], P[k], *t_now);
            ro_p[0] = ro[0]; // 0.000001;
            p_p[0] = p[0];
            //printf("Omni,   %lf,   %lf,   %lf,  %d \n", T[k], *t_now, T[k - 1], *dev_mas_Omni);
            break;
        }
    }

}

__global__ void takeVoyadger(double* T_do, double* t_now, double* T_V, int* dev_mas_V, int* voy)
{
    for (int k = *dev_mas_V; k < data_voy; k++)
    {
        if (T_V[k] > *t_now)
        {
            *dev_mas_V = k;
            break;
        }
    }

    if (*T_do > T_V[*dev_mas_V] - *t_now)
    {
        *T_do = T_V[*dev_mas_V] - *t_now;
        *voy = 1;
    }
    else
    {
        *voy = 0;
    }

}

__global__ void takeVoyadger2(double* ro, double* ro2, double* p, double* u, double* distV, int* dev_mas_V, double* j1, double* j2, double* j3)
{
    int kk = int((distV[*dev_mas_V] - 1.0) / dx);
    double r1 = L + kk * dx;
    double r2 = L + (kk + 1) * dx;
    //*j1 = linear(r1, ro[kk] - ro2[kk], r2, ro[kk + 1] - ro2[kk + 1], distV[*dev_mas_V]);
    *j1 = linear(r1, ro2[kk], r2, ro2[kk + 1], distV[*dev_mas_V]);
    *j2 = linear(r1, u[kk], r2, u[kk + 1], distV[*dev_mas_V]);
    *j3 = linear(r1, p[kk], r2, p[kk + 1], distV[*dev_mas_V]);
}

__global__ void takeVoyadger2_2(double* ro, double* ro2, double* p, double* u, double* v, double* w, double* distV, int* dev_mas_V, double* j1, double* j2, double* j3)
{
    // ??? ?????????? ????????
    int kk = int((distV[*dev_mas_V] - 1.0) / dx);
    double r1 = L + kk * dx;
    double r2 = L + (kk + 1) * dx;
    double a1, a2, a3;
    //*j1 = linear(r1, ro[kk] - ro2[kk], r2, ro[kk + 1] - ro2[kk + 1], distV[*dev_mas_V]);
    *j1 = linear(r1, ro2[kk], r2, ro2[kk + 1], distV[*dev_mas_V]);
    a1 = linear(r1, u[kk], r2, u[kk + 1], distV[*dev_mas_V]);
    a2 = linear(r1, v[kk], r2, v[kk + 1], distV[*dev_mas_V]);
    a3 = linear(r1, w[kk], r2, w[kk + 1], distV[*dev_mas_V]);
    *j2 = sqrt(kv(a1) + kv(a2) + kv(a3));
    *j3 = linear(r1, p[kk], r2, p[kk + 1], distV[*dev_mas_V]);
}

__global__ void init_time(double* T_do, double* T)
{
    *T_do = 0.0001 * minut;
    *T = 5.0 * minut;
}

__global__ void calcul(double* ro, double* u, double* p, double* ro2, double* u2, double* p2,//
    double* T, double* T_do, int* N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // ?????????? ?????? ??????? ?????? (???????? ??????)
    
    if (i >= *N)
    {
        return;
    }

    if (i == 0) // Ƹ????? ????????? ???????
    {
        return;
    }

    if (i == *N - 1) // Ƹ????? ????????? ???????
    {
        ro2[*N - 1] = ro[*N - 2];
        p2[*N - 1] = p[*N - 2];
        u2[*N - 1] = u[*N - 2];
        return;
    }

    double P[5] = { 0.0 };
    double r = L + i * dx;
    double B1 = 0.0;
    double B2 = 0.0;
    double B3 = 0.0;
    double time2 = 0.1 * minut;
    double ro1 = ro[i];
    double p1 = p[i];
    double u1 = u[i];
    double ro3 = ro[i + 1];
    double p3 = p[i + 1];
    double u3 = u[i + 1];
    double ro4 = ro[i - 1];
    double p4 = p[i - 1];
    double u4 = u[i - 1];

    double roL = ro1;
    double pL = p1;
    double uL = u1;
    double roR = ro3;
    double pR = p3;
    double uR = u3;
    double PQ;

    if (i > 1 && i < *N - 2)
    {
        roL = linear(r - dx, ro4, r, ro1, r + dx, ro3, r + dx / 2.0);
        pL = linear(r - dx, p4, r, p1, r + dx, p3, r + dx / 2.0);
        uL = linear(r - dx, u4, r, u1, r + dx, u3, r + dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }

        roR = linear(r, ro1, r + dx, ro3, r + 2.0 * dx, ro[i + 2], r + dx / 2.0);
        pR = linear(r, p1, r + dx, p3, r + 2.0 * dx, p[i + 2], r + dx / 2.0);
        uR = linear(r, u1, r + dx, u3, r + 2.0 * dx, u[i + 2], r + dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro3;
        }
        if (pR <= 0.0)
        {
            pR = p3;
        }

    }

    double CC, RO;
    time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, 1.0, pL, uL, 0.0, 1.0, roR, 1.0, pR, uR, 0.0, 1.0, 0.0, P, PQ, 1.0, 0.0, dx, CC, RO));
    B1 = P[0];
    B2 = P[1];
    B3 = P[3];  // 4

    roL = ro1;
    pL = p1;
    uL = u1;
    roR = ro4;
    pR = p4;
    uR = u4;

    if (i > 1 && i < *N - 2)
    {
        roL = linear(r - dx, ro4, r, ro1, r + dx, ro3, r - dx / 2.0);
        pL = linear(r - dx, p4, r, p1, r + dx, p3, r - dx / 2.0);
        uL = linear(r - dx, u4, r, u1, r + dx, u3, r - dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }

        roR = linear(r - 2.0 * dx, ro[i - 2], r - dx, ro4, r, ro1, r - dx / 2.0);
        pR = linear(r - 2.0 * dx, p[i - 2], r - dx, p4, r, p1, r - dx / 2.0);
        uR = linear(r - 2.0 * dx, u[i - 2], r - dx, u4, r, u1, r - dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro4;
        }
        if (pR <= 0.0)
        {
            pR = p4;
        }

    }


    time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, 1.0, pL, uL, 0.0, 1.0, roR, 1.0, pR, uR, 0.0, 1.0, 0.0, P, PQ, -1.0, 0.0, dx, CC, RO));

    B1 = B1 + P[0];
    B2 = B2 + P[1];
    B3 = B3 + P[3];  // 4

    ro2[i] = -*T_do * (B1 / dx - (2.0 / r) * (-ro1 * u1)) + ro1;
    if (ro2[i] <= 0.0)
    {
        printf("Error ro \n");
    }
    u2[i] = (-*T_do * (B2 / dx + Kn * n_H * exp(-r0/r) * kv(u1) * ro1 * sigma(u1) - (2.0 / r) * (-ro1 * u1 * u1)) + ro1 * u1) / ro2[i];
    //p2[i] = ((-*T_do * (B3 / dx + 0.5 * Kn * n_H * exp(-r0 / r) * kv(u1) * u1 * ro1 * sigma(u1) - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * u1 * u1 * u1 / 2.0)) + //
    //    p1 / (ggg - 1.0) + ro1 * u1 * u1 / 2.0) - ro2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);
    //u2[i] = (-*T_do * (B2 / dx - (2.0 / r) * (-ro1 * u1 * u1)) + ro1 * u1) / ro2[i];
    //p2[i] = ((-*T_do * (B3 / dx - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * u1 * u1 * u1 / 2.0)) + //
    //    p1 / (ggg - 1.0) + ro1 * u1 * u1 / 2.0) - ro2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);
    p2[i] = ((-*T_do * (B3 / dx + Kn * n_H * exp(-r0 / r) * kv(u1) * u1 * ro1 * sigma(u1) - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * u1 * u1 * u1 / 2.0)) + //
        p1 / (ggg - 1.0) + ro1 * u1 * u1 / 2.0) - ro2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);

    if (p2[i] <= 0.0)
    {
        p2[i] = 0.000001;
    }

    if (time2 < *T)
    {
        *T = time2;
    }

}

__global__ void calcul_component(double* ro, double* u, double* p, double* ro2, double* u2, double* p2, double* ro_p, double* ro_p2, double* p_p, double* p_p2,//
    double* T, double* T_do, int* N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // ?????????? ?????? ??????? ?????? (???????? ??????)

    if (i >= *N)
    {
        return;
    }

    if (i == 0) // Ƹ????? ????????? ???????
    {
        ro2[0] = ro[0];
        p2[0] = p[0];
        u2[0] = u[0];
        ro_p2[0] = ro_p[0];
        p_p2[0] = p_p[0];
        return;
    }

    if (i == *N - 1) // ?????? ????????? ???????
    {
        ro2[*N - 1] = ro[*N - 2];
        p2[*N - 1] = p[*N - 2];
        u2[*N - 1] = u[*N - 2];

        ro_p2[*N - 1] = ro_p[*N - 2];
        p_p2[*N - 1] = p_p[*N - 2];
        return;
    }

    double P[5] = { 0.0 };
    double r = L + i * dx;
    double B1 = 0.0;
    double B2 = 0.0;
    double B3 = 0.0;
    double Bp1 = 0.0;
    double Bp2 = 0.0;
    double Bp3 = 0.0;
    double B_pi1 = 0.0;
    double B_p3 = 0.0;
    double time2 = 0.3 * minut;  // 0.3
    double ro1 = ro[i];
    double p1 = p[i];
    double ro_p1 = ro_p[i];
    double p_p1 = p_p[i];
    double u1 = u[i];
    double ro3 = ro[i + 1];
    double p3 = p[i + 1];
    double ro_p3 = ro_p[i + 1];
    double p_p3 = p_p[i + 1];
    double u3 = u[i + 1];
    double ro4 = ro[i - 1];
    double p4 = p[i - 1];
    double ro_p4 = ro_p[i - 1];
    double p_p4 = p_p[i - 1];
    double u4 = u[i - 1];

    double roL = ro1;
    double pL = p1;
    double ro_pL = ro_p1;
    double p_pL = p_p1;
    double uL = u1;
    double roR = ro3;
    double pR = p3;
    double ro_pR = ro_p3;
    double p_pR = p_p3;
    double uR = u3;
    double PQ;

    if (i > 1 && i < *N - 2)
    {
        roL =   linear(r - dx, ro4,   r, ro1,   r + dx, ro3,   r + dx / 2.0);
        pL =    linear(r - dx, p4,    r, p1,    r + dx, p3,    r + dx / 2.0);
        ro_pL = linear(r - dx, ro_p4, r, ro_p1, r + dx, ro_p3, r + dx / 2.0);
        p_pL =  linear(r - dx, p_p4,  r, p_p1,  r + dx, p_p3,  r + dx / 2.0);
        uL =    linear(r - dx, u4,    r, u1,    r + dx, u3,    r + dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }
        if (ro_pL <= 0.0)
        {
            ro_pL = ro_p1;
        }
        if (p_pL <= 0.0)
        {
            p_pL = p_p1;
        }

        roR =   linear(r, ro1,   r + dx, ro3,   r + 2.0 * dx, ro[i + 2],    r + dx / 2.0);
        pR =    linear(r, p1,    r + dx, p3,    r + 2.0 * dx, p[i + 2],     r + dx / 2.0);
        ro_pR = linear(r, ro_p1, r + dx, ro_p3, r + 2.0 * dx, ro_p[i + 2],  r + dx / 2.0);
        p_pR =  linear(r, p_p1,  r + dx, p_p3,  r + 2.0 * dx, p_p[i + 2],   r + dx / 2.0);
        uR =    linear(r, u1,    r + dx, u3,    r + 2.0 * dx, u[i + 2],     r + dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro3;
        }
        if (pR <= 0.0)
        {
            pR = p3;
        }
        if (ro_pR <= 0.0)
        {
            ro_pR = ro_p3;
        }
        if (p_pR <= 0.0)
        {
            p_pR = p_p3;
        }

    }

    double PP, RO; 
    time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, ro_pL, pL, uL, 0.0, p_pL, roR, ro_pR, pR, uR, 0.0, p_pR, 0.0, P, PQ, 1.0, 0.0, dx, RO, PP));
    B1 = P[0];
    B2 = P[1];
    B3 = P[3];  // 4

    time2 = min(time2, HLLC_2d_Korolkov_b_s(ro_pL, 1.0, p_pL, uL, 0.0, 1.0, ro_pR, 1.0, p_pR, uR, 0.0, 1.0, 0.0, P, PQ, 1.0, 0.0, dx, RO, PP));
    Bp1 = P[0];
    Bp2 = P[1];
    Bp3 = P[3];  // 4

    B_pi1 = PQ;
    B_p3 = PP/pow(RO, ggg);// P[4];

    roL = ro1;
    pL = p1;
    ro_pL = ro_p1;
    p_pL = p_p1;
    uL = u1;
    roR = ro4;
    pR = p4;
    ro_pR = ro_p4;
    p_pR = p_p4;
    uR = u4;

    if (i > 1 && i < *N - 2)
    {
        roL =   linear(r - dx, ro4,   r, ro1,   r + dx, ro3,   r - dx / 2.0);
        pL =    linear(r - dx, p4,    r, p1,    r + dx, p3,    r - dx / 2.0);
        ro_pL = linear(r - dx, ro_p4, r, ro_p1, r + dx, ro_p3, r - dx / 2.0);
        p_pL =  linear(r - dx, p_p4,  r, p_p1,  r + dx, p_p3,  r - dx / 2.0);
        uL =    linear(r - dx, u4,    r, u1,    r + dx, u3,    r - dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }
        if (ro_pL <= 0.0)
        {
            ro_pL = ro_p1;
        }
        if (p_pL <= 0.0)
        {
            p_pL = p_p1;
        }

        roR =   linear(r - 2.0 * dx, ro[i - 2],   r - dx, ro4,   r, ro1,   r - dx / 2.0);
        pR =    linear(r - 2.0 * dx, p[i - 2],    r - dx, p4,    r, p1,    r - dx / 2.0);
        ro_pR = linear(r - 2.0 * dx, ro_p[i - 2], r - dx, ro_p4, r, ro_p1, r - dx / 2.0);
        p_pR =  linear(r - 2.0 * dx, p_p[i - 2],  r - dx, p_p4,  r, p_p1,  r - dx / 2.0);
        uR =    linear(r - 2.0 * dx, u[i - 2],    r - dx, u4,    r, u1,    r - dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro4;
        }
        if (pR <= 0.0)
        {
            pR = p4;
        }
        if (ro_pR <= 0.0)
        {
            ro_pR = ro_p4;
        }
        if (p_pR <= 0.0)
        {
            p_pR = p_p4;
        }

    }

    PP = 0.0;
    RO = 0.0;
    time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, ro_pL, pL, uL, 0.0, p_pL, roR, ro_pR, pR, uR, 0.0, p_pR, 0.0, P, PQ, -1.0, 0.0, dx, RO, PP));

    B1 = B1 + P[0];
    B2 = B2 + P[1];
    B3 = B3 + P[3];  // 4

    time2 = min(time2, HLLC_2d_Korolkov_b_s(ro_pL, 1.0, p_pL, uL, 0.0, 1.0, ro_pR, 1.0, p_pR, uR, 0.0, 1.0, 0.0, P, PQ, -1.0, 0.0, dx, RO, PP));

    Bp1 = Bp1 + P[0];
    Bp2 = Bp2 + P[1];
    Bp3 = Bp3 + P[3];

    B_pi1 = B_pi1 + PQ;

    B_p3 = B_p3 - PP / pow(RO, ggg);// P[4];

    //B_pi1 = (ro_p1 * u1 - ro_p4 * u4);

    //ro2[i] = -*T_do * (B1 / dx - (2.0 / r) * (-ro1 * u1)) + ro1;
    //if (ro2[i] <= 0.0)
    //{
    //    printf("Error ro \n");
    //}
    //u2[i] = (-*T_do * (B2 / dx + Kn * n_H * exp(-r0 / r) * kv(u1) * ro1 * sigma(u1) - (2.0 / r) * (-ro1 * u1 * u1)) + ro1 * u1) / ro2[i];
    //u2[i] = (-*T_do * (B2 / dx - (2.0 / r) * (-ro1 * u1 * u1)) + ro1 * u1) / ro2[i];
    //p2[i] = ((-*T_do * (B3 / dx - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * u1 * u1 * u1 / 2.0)) + //
    //    p1 / (ggg - 1.0) + ro1 * u1 * u1 / 2.0) - ro2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);
    
    //p2[i] = ((-*T_do * (B3 / dx + Kn * n_H * exp(-r0 / r) * kv(u1) * u1 * ro1 * sigma(u1) - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * u1 * u1 * u1 / 2.0)) + //
    //    p1 / (ggg - 1.0) + ro1 * u1 * u1 / 2.0) - ro2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);


    // ??????? ????????? 2-??????

    double pp = p_p1;
    double roo = ro_p1;

    double roH = n_H * exp(-r0 / r);
    double UH = sqrt(kv(v_H - u1) + 4.0 / pi * (kv(c_H) + pp / roo));
    double UMH = sqrt(kv(v_H - u1) + 64.0 / (9.0 * pi) * (kv(c_H) + pp / roo));
    double nu = Kn * roo * roH * UMH * sigma(UMH);
    double Q1, Q2, Q3, Q1pi, Q3p;
    Q1 = 0.0;
    Q2 = nu* (v_H - u1);
    Q3 = nu* (0.5 * (kv(v_H) - kv(u1)) + (kv(c_H) - pp / roo) * UH / UMH);
    Q1pi = -Kn * roo * roH * UH * sigma(UH);// +0.0 * nu * (0.5 * (kv(v_H - u1)) + (kv(c_H) - p_pp / ro_pp) * UH / UMH);
    Q3p = 0.02 * nu * (0.5 * (kv(v_H - u1)) + (kv(c_H) - pp / roo) * UH / UMH);;

    //printf("%lf, %lf\n", Q3, Q3p);

    ro2[i] = -*T_do * (B1 / dx - Q1 - (2.0 / r) * (-ro1 * u1)) + ro1;
    u2[i] = (-*T_do * (B2 / dx - Q2 - (2.0 / r) * (-ro1 * u1 * u1)) + ro1 * u1) / ro2[i];
    p2[i] = ((-*T_do * (B3 / dx - Q3 - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * u1 * u1 * u1 / 2.0)) + //
        p1 / (ggg - 1.0) + ro1 * u1 * u1 / 2.0) - ro2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);


    ro_p2[i] = -*T_do * (Bp1 / dx - Q1pi - (2.0 / r) * (-ro_p1 * u1)) + ro_p1;
    double Q = (ro_p2[i] * u2[i] - ro_p1 * u1) / *T_do + Bp2 / dx - (2.0 / r) * (-ro_p1 * u1 * u1);
    p_p2[i] = ((-*T_do * (Bp3 / dx - (Q3p - Q1pi * (0.5 * kv(u1) + p_p1 / g1) + Q * u1) - (2.0 / r) * (-(ggg * p_p1 * u1) / (ggg - 1.0) - ro_p1 * u1 * u1 * u1 / 2.0)) + //
        p_p1 / (ggg - 1.0) + ro_p1 * u1 * u1 / 2.0) - ro_p2[i] * u2[i] * u2[i] / 2.0) * (ggg - 1.0);
    

    //ro_p2[i] = ro_p1; // -*T_do * (B_pi1 / dx - Q1pi - (2.0 / r) * (-ro_p1 * u1)) + ro_p1; // ????? ????????? ?????-?????

    //u2[i] = (-*T_do * (B_p2 / dx - Q2 - (2.0 / r) * (-ro_p1 * u1 * u1)) + ro_p1 * u1) / ro_p2[i];
    
    //p_p2[i] = ((-*T_do * (B_p3 / dx - Q3p - (2.0 / r) * (-(ggg * p_p1 * u1) / (ggg - 1.0) - (ro1 - ro_p1) *u1 * u1 * u1 / 2.0)) + //
    //    p_p1 / (ggg - 1.0) + (ro1 - ro_p1) * u1 * u1 / 2.0) - (ro2[i] - ro_p2[i]) * u2[i] * u2[i] / 2.0) * (ggg - 1.0);

    //p_p2[i] = *T_do * (ggg - 1.0) *  (-(1.0 / (ggg - 1.0)) * (p_p1 * u1 - p_p4 * u4) / dx - (ggg / (ggg - 1.0)) * (2.0 * p_p1 * u1 / r) - p_p1 * (u1 - u4) / dx) + p_p1;

    //p_p2[i] = p_p1;// (-*T_do * u1 * (B_p3) / dx + p_p1 / pow(ro1 - ro_p1, ggg))* pow(ro2[i] - ro_p2[i], ggg);


    if (p_p2[i] < 0.0)
    {
        printf("ERROR  707  p < 0 = %lf __  %lf \n", r, p_p2[i]);
        p_p2[i] = p2[i];
    }

    if (ro_p2[i] < 0.0)
    {
        printf("ERROR  707  ro < 0 = %lf __  %lf \n", r, ro_p2[i]);
        ro_p2[i] = ro2[i];
    }

    if (time2 < *T)
    {
        *T = time2;
    }

}

__global__ void calcul_component_2(double* ro, double* u, double* v, double* w, double* p, double* ro2, double* u2, double* v2, double* w2, double* p2, double* ro_p, double* ro_p2, double* p_p, double* p_p2,//
    double* T, double* T_do, int* N, int* dev_mas_V, bool* boo, double kk = 1.0)
{
    // ? ??????????????? ???????????? ????????
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // ?????????? ?????? ??????? ?????? (???????? ??????)

    if (i >= *N)
    {
        return;
    }

    if (i == 0) // Ƹ????? ????????? ???????
    {
        ro2[0] = ro[0];
        p2[0] = p[0];
        u2[0] = u[0];
        v2[0] = v[0];
        w2[0] = w[0];
        ro_p2[0] = ro_p[0];
        p_p2[0] = p_p[0];
        return;
    }

    if (i == *N - 1) // ?????? ????????? ???????
    {
        ro2[*N - 1] = ro[*N - 2];
        p2[*N - 1] = p[*N - 2];
        u2[*N - 1] = u[*N - 2];
        v2[*N - 1] = v[*N - 2];
        w2[*N - 1] = w[*N - 2];

        ro_p2[*N - 1] = ro_p[*N - 2];
        p_p2[*N - 1] = p_p[*N - 2];
        return;
    }

    double P[8] = { 0.0 };
    double r = L + i * dx;
    double B1 = 0.0;
    double B2 = 0.0;
    double B3 = 0.0;
    double B4 = 0.0;
    double B5 = 0.0;
    double Bp1 = 0.0;
    double Bp2 = 0.0;
    double Bp3 = 0.0;
    double Bp4 = 0.0;
    double Bp5 = 0.0;
    double B_pi1 = 0.0;
    double B_p3 = 0.0;
    double time2 = 0.5 * minut/kk;
    /*if (*dev_mas_V < 4000)
    {
        time2 = 1.0 * minut;
    }
    else if (*dev_mas_V < 6000)
    {
        time2 = 0.3 * minut;
    }
    else if (*dev_mas_V < 7500)
    {
        time2 = 1.0 * minut;
    }
    else if (*dev_mas_V < 8500)
    {
        time2 = 0.3 * minut;
    }
    else
    {
        time2 = 1.0 * minut;
    }*/

    double ro1 = ro[i];
    double p1 = p[i];
    double ro_p1 = ro_p[i];
    double p_p1 = p_p[i];
    double u1 = u[i];
    double v1 = v[i];
    double w1 = w[i];
    double ro3 = ro[i + 1];
    double p3 = p[i + 1];
    double ro_p3 = ro_p[i + 1];
    double p_p3 = p_p[i + 1];
    double u3 = u[i + 1];
    double v3 = v[i + 1];
    double w3 = w[i + 1];
    double ro4 = ro[i - 1];
    double p4 = p[i - 1];
    double ro_p4 = ro_p[i - 1];
    double p_p4 = p_p[i - 1];
    double u4 = u[i - 1];
    double v4 = v[i - 1];
    double w4 = w[i - 1];

    double roL = ro1;
    double pL = p1;
    double ro_pL = ro_p1;
    double p_pL = p_p1;
    double uL = u1;
    double vL = v1;
    double wL = w1;
    double roR = ro3;
    double pR = p3;
    double ro_pR = ro_p3;
    double p_pR = p_p3;
    double uR = u3;
    double vR = v3;
    double wR = w3;
    double PQ;

    if (i > 1 && i < *N - 2)
    {
        roL = linear(r - dx, ro4, r, ro1, r + dx, ro3, r + dx / 2.0);
        pL = linear(r - dx, p4, r, p1, r + dx, p3, r + dx / 2.0);
        ro_pL = linear(r - dx, ro_p4, r, ro_p1, r + dx, ro_p3, r + dx / 2.0);
        p_pL = linear(r - dx, p_p4, r, p_p1, r + dx, p_p3, r + dx / 2.0);
        uL = linear(r - dx, u4, r, u1, r + dx, u3, r + dx / 2.0);
        vL = linear(r - dx, v4, r, v1, r + dx, v3, r + dx / 2.0);
        wL = linear(r - dx, w4, r, w1, r + dx, w3, r + dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }
        if (ro_pL <= 0.0)
        {
            ro_pL = ro_p1;
        }
        if (p_pL <= 0.0)
        {
            p_pL = p_p1;
        }

        roR = linear(r, ro1, r + dx, ro3, r + 2.0 * dx, ro[i + 2], r + dx / 2.0);
        pR = linear(r, p1, r + dx, p3, r + 2.0 * dx, p[i + 2], r + dx / 2.0);
        ro_pR = linear(r, ro_p1, r + dx, ro_p3, r + 2.0 * dx, ro_p[i + 2], r + dx / 2.0);
        p_pR = linear(r, p_p1, r + dx, p_p3, r + 2.0 * dx, p_p[i + 2], r + dx / 2.0);
        uR = linear(r, u1, r + dx, u3, r + 2.0 * dx, u[i + 2], r + dx / 2.0);
        vR = linear(r, v1, r + dx, v3, r + 2.0 * dx, v[i + 2], r + dx / 2.0);
        wR = linear(r, w1, r + dx, w3, r + 2.0 * dx, w[i + 2], r + dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro3;
        }
        if (pR <= 0.0)
        {
            pR = p3;
        }
        if (ro_pR <= 0.0)
        {
            ro_pR = ro_p3;
        }
        if (p_pR <= 0.0)
        {
            p_pR = p_p3;
        }

    }

    double PP, RO;
    //time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, ro_pL, pL, uL, 0.0, p_pL, roR, ro_pR, pR, uR, 0.0, p_pR, 0.0, P, PQ, 1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(roL, 1.0, pL, uL, vL, wL, 0.0, 0.0, 0.0, //
        roR, 1.0, pR, uR, vR, wR, 0.0, 0.0, 0.0, P, PQ, 1.0, 0.0, 0.0, dx, 2));
    B1 = P[0];
    B2 = P[1];
    B3 = P[2];
    B4 = P[3];
    B5 = P[7];  // 4

    //time2 = min(time2, HLLC_2d_Korolkov_b_s(ro_pL, 1.0, p_pL, uL, 0.0, 1.0, ro_pR, 1.0, p_pR, uR, 0.0, 1.0, 0.0, P, PQ, 1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(ro_pL, 1.0, p_pL, uL, vL, wL, 0.0, 0.0, 0.0, //
        ro_pR, 1.0, p_pR, uR, vR, wR, 0.0, 0.0, 0.0, P, PQ, 1.0, 0.0, 0.0, dx, 2));
    Bp1 = P[0];
    Bp2 = P[1];
    Bp3 = P[2];
    Bp4 = P[3];
    Bp5 = P[7];  // 4

    roL = ro1;
    pL = p1;
    ro_pL = ro_p1;
    p_pL = p_p1;
    uL = u1;
    vL = v1;
    wL = w1;
    roR = ro4;
    pR = p4;
    ro_pR = ro_p4;
    p_pR = p_p4;
    uR = u4;
    vR = v4;
    wR = w4;

    if (i > 1 && i < *N - 2)
    {
        roL = linear(r - dx, ro4, r, ro1, r + dx, ro3, r - dx / 2.0);
        pL = linear(r - dx, p4, r, p1, r + dx, p3, r - dx / 2.0);
        ro_pL = linear(r - dx, ro_p4, r, ro_p1, r + dx, ro_p3, r - dx / 2.0);
        p_pL = linear(r - dx, p_p4, r, p_p1, r + dx, p_p3, r - dx / 2.0);
        uL = linear(r - dx, u4, r, u1, r + dx, u3, r - dx / 2.0);
        vL = linear(r - dx, v4, r, v1, r + dx, v3, r - dx / 2.0);
        wL = linear(r - dx, w4, r, w1, r + dx, w3, r - dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }
        if (ro_pL <= 0.0)
        {
            ro_pL = ro_p1;
        }
        if (p_pL <= 0.0)
        {
            p_pL = p_p1;
        }

        roR = linear(r - 2.0 * dx, ro[i - 2], r - dx, ro4, r, ro1, r - dx / 2.0);
        pR = linear(r - 2.0 * dx, p[i - 2], r - dx, p4, r, p1, r - dx / 2.0);
        ro_pR = linear(r - 2.0 * dx, ro_p[i - 2], r - dx, ro_p4, r, ro_p1, r - dx / 2.0);
        p_pR = linear(r - 2.0 * dx, p_p[i - 2], r - dx, p_p4, r, p_p1, r - dx / 2.0);
        uR = linear(r - 2.0 * dx, u[i - 2], r - dx, u4, r, u1, r - dx / 2.0);
        vR = linear(r - 2.0 * dx, v[i - 2], r - dx, v4, r, v1, r - dx / 2.0);
        wR = linear(r - 2.0 * dx, w[i - 2], r - dx, w4, r, w1, r - dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro4;
        }
        if (pR <= 0.0)
        {
            pR = p4;
        }
        if (ro_pR <= 0.0)
        {
            ro_pR = ro_p4;
        }
        if (p_pR <= 0.0)
        {
            p_pR = p_p4;
        }

    }

    PP = 0.0;
    RO = 0.0;
    //time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, ro_pL, pL, uL, 0.0, p_pL, roR, ro_pR, pR, uR, 0.0, p_pR, 0.0, P, PQ, -1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(roL, 1.0, pL, uL, vL, wL, 0.0, 0.0, 0.0, //
        roR, 1.0, pR, uR, vR, wR, 0.0, 0.0, 0.0, P, PQ, -1.0, 0.0, 0.0, dx, 2));

    B1 = B1 + P[0];
    B2 = B2 + P[1];
    B3 = B3 + P[2];
    B4 = B4 + P[3];
    B5 = B5 + P[7];  // 4

    //time2 = min(time2, HLLC_2d_Korolkov_b_s(ro_pL, 1.0, p_pL, uL, 0.0, 1.0, ro_pR, 1.0, p_pR, uR, 0.0, 1.0, 0.0, P, PQ, -1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(ro_pL, 1.0, p_pL, uL, vL, wL, 0.0, 0.0, 0.0, //
        ro_pR, 1.0, p_pR, uR, vR, wR, 0.0, 0.0, 0.0, P, PQ, -1.0, 0.0, 0.0, dx, 2));

    Bp1 = Bp1 + P[0];
    Bp2 = Bp2 + P[1];
    Bp3 = Bp3 + P[2];
    Bp4 = Bp4 + P[3];
    Bp5 = Bp5 + P[7];


    // ??????? ????????? 2-??????

    double pp = p_p1;
    double roo = ro_p1;

    double roH = n_H * exp(-r0 / r);
    double UH = sqrt(kv(v_H - u1) + kv(v1) + kv(w1) + 4.0 / pi * (kv(c_H) + pp / roo));
    double UMH = sqrt(kv(v_H - u1) + kv(v1) + kv(w1) + 64.0 / (9.0 * pi) * (kv(c_H) + pp / roo));
    double nu = Kn * roo * roH * UMH * sigma(UMH);
    double Q1, Q21, Q22, Q23, Q3, Q1pi, Q3p;
    Q1 = 0.0;//
    Q21 = 0.0;//nu * (v_H - u1);
    Q22 = 0.0;//nu * (0.0 - v1);
    Q23 = 0.0;//nu * (0.0 - w1);
    Q3 = 0.0;//nu * (0.5 * (kv(v_H) - kv(u1) - kv(v1) - kv(w1)) + (kv(c_H) - pp / roo) * UH / UMH);
    Q1pi = 0.0;//-Kn * roo * roH * UH * sigma(UH);// +0.0 * nu * (0.5 * (kv(v_H - u1)) + (kv(c_H) - p_pp / ro_pp) * UH / UMH);
    Q3p = 0.0;//0.02 * nu * (0.5 * (kv(v_H - u1) + kv(v1) + kv(w1)) + (kv(c_H) - pp / roo) * UH / UMH);;

    //printf("%lf, %lf\n", Q3, Q3p);

    ro2[i] = -*T_do * (B1 / dx - Q1 - (2.0 / r) * (-ro1 * u1)) + ro1;
    u2[i] = (-*T_do * (B2 / dx - Q21 - (2.0 / r) * (-ro1 * u1 * u1 + 0.5 * ro1 * (kv(v1) + kv(w1)))) + ro1 * u1) / ro2[i];
    v2[i] = (-*T_do * (B3 / dx - Q22 - (3.0 / r) * (-ro1 * u1 * v1)) + ro1 * v1) / ro2[i];
    w2[i] = (-*T_do * (B4 / dx - Q23 - (3.0 / r) * (-ro1 * u1 * w1)) + ro1 * w1) / ro2[i];

    //double uu = kv(u1) + kv(v1) + kv(w1);
    //double uu2 = kv(u2[i]) + kv(v2[i]) + kv(w2[i]);

    //p2[i] = (-*T_do * (B5 / dx - Q3 - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * uu * u1 / 2.0)) + //
    //     (ro1 * uu / 2.0 - ro2[i] * uu2 / 2.0) + p1 / (ggg - 1.0) ) * (ggg - 1.0);

    p2[i] = (-*T_do * (B5 / dx - Q3 - (2.0 / r) * (-(ggg * p1 * u1) / (ggg - 1.0) - ro1 * (kv(u1) + kv(v1) + kv(w1)) * u1 / 2.0)) + //
        (ro1 * (kv(u1) + kv(v1) + kv(w1)) / 2.0 - ro2[i] * (kv(u2[i]) + kv(v2[i]) + kv(w2[i])) / 2.0) + p1 / (ggg - 1.0)) * (ggg - 1.0);


    ro_p2[i] = ro2[i];// -*T_do * (Bp1 / dx - Q1pi - (2.0 / r) * (-ro_p1 * u1)) + ro_p1;
    //double M1, M2, M3;

    //M1 = (ro_p2[i] * u2[i] - ro_p1 * u1) / *T_do + Bp2 / dx - (2.0 / r) * (-ro_p1 * u1 * u1 + 0.5 * ro_p1 * (kv(v1) + kv(w1)));
    //M2 = (ro_p2[i] * v2[i] - ro_p1 * v1) / *T_do + Bp3 / dx - (3.0 / r) * (-ro_p1 * u1 * v1);
    //M3 = (ro_p2[i] * w2[i] - ro_p1 * w1) / *T_do + Bp4 / dx - (3.0 / r) * (-ro_p1 * u1 * w1);

    //double sk = M1 * u1 + M2 * v1 + M3 * w1;

    p_p2[i] = p2[i]; // ((-*T_do * (Bp5 / dx - (Q3p - Q1pi * (0.5 * uu + p_p1 / g1) + sk) - (2.0 / r) * (-(ggg * p_p1 * u1) / (ggg - 1.0) - ro_p1 * uu * u1 / 2.0)) + //
       // p_p1 / (ggg - 1.0) + ro_p1 * uu / 2.0) - ro_p2[i] * uu2 / 2.0) * (ggg - 1.0);



    if (p2[i] < 0.0)
    {
        printf("ERROR  707  p < 0 = %lf  __  %lf  __  %lf \n", r, p2[i], *T_do);
        p2[i] = 0.0000001;
        p_p2[i] = 0.0000001;
        time2 = 0.1 * time2;
        if (kk < 30.0)
        {
            *boo = false;
        }
    }

    if (ro_p2[i] < 0.0)
    {
        printf("ERROR  707  ro < 0 = %lf __  %lf \n", r, ro_p2[i]);
        ro_p2[i] = ro2[i];
    }

    if (time2 < *T)
    {
        *T = time2;
    }

}


__global__ void calcul_component_mgd(double* ro, double* u, double* p, double* ro2, double* u2, double* p2, double* ro_p, double* ro_p2, double* p_p, double* p_p2,//
    double* by, double* by2, double* T, double* T_do, int* N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // ?????????? ?????? ??????? ?????? (???????? ??????)

    if (i >= *N)
    {
        return;
    }

    if (i == 0) // Ƹ????? ????????? ???????
    {
        ro2[0] = ro[0];
        p2[0] = p[0];
        u2[0] = u[0];
        ro_p2[0] = ro_p[0];
        p_p2[0] = p_p[0];
        by2[0] = by[0];
        return;
    }

    if (i == *N - 1) // ?????? ????????? ???????
    {
        ro2[*N - 1] = ro[*N - 2];
        p2[*N - 1] = p[*N - 2];
        u2[*N - 1] = u[*N - 2];

        ro_p2[*N - 1] = ro_p[*N - 2];
        p_p2[*N - 1] = p_p[*N - 2];
        by2[*N - 1] = by[*N - 2];
        return;
    }

    double P[8] = { 0.0 };
    double r = L + i * dx;
    double B1 = 0.0;
    double B2 = 0.0;
    double B3 = 0.0;
    double B4 = 0.0;
    double Bp1 = 0.0;
    double Bp2 = 0.0;
    double Bp3 = 0.0;
    double Bp4 = 0.0;
    double B_pi1 = 0.0;
    double B_p3 = 0.0;
    double time2 = 0.3 * minut;  // 0.3
    double ro1 = ro[i];
    double p1 = p[i];
    double ro_p1 = ro_p[i];
    double p_p1 = p_p[i];
    double u1 = u[i];
    double ro3 = ro[i + 1];
    double p3 = p[i + 1];
    double ro_p3 = ro_p[i + 1];
    double p_p3 = p_p[i + 1];
    double u3 = u[i + 1];
    double ro4 = ro[i - 1];
    double p4 = p[i - 1];
    double ro_p4 = ro_p[i - 1];
    double p_p4 = p_p[i - 1];
    double u4 = u[i - 1];

    //printf("%lf  __  %lf \n", ro1 * u1 * u1/kv(br(r + dx/2.0)), ro3 * u3 * u3 / kv(br(r + dx / 2.0)));

    double roL = ro1;
    double pL = p1;
    double ro_pL = ro_p1;
    double p_pL = p_p1;
    double uL = u1;
    double roR = ro3;
    double pR = p3;
    double ro_pR = ro_p3;
    double p_pR = p_p3;
    double uR = u3;
    double PQ;

    if (i > 1 && i < *N - 2)
    {
        roL = linear(r - dx, ro4, r, ro1, r + dx, ro3, r + dx / 2.0);
        pL = linear(r - dx, p4, r, p1, r + dx, p3, r + dx / 2.0);
        ro_pL = linear(r - dx, ro_p4, r, ro_p1, r + dx, ro_p3, r + dx / 2.0);
        p_pL = linear(r - dx, p_p4, r, p_p1, r + dx, p_p3, r + dx / 2.0);
        uL = linear(r - dx, u4, r, u1, r + dx, u3, r + dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }
        if (ro_pL <= 0.0)
        {
            ro_pL = ro_p1;
        }
        if (p_pL <= 0.0)
        {
            p_pL = p_p1;
        }

        roR = linear(r, ro1, r + dx, ro3, r + 2.0 * dx, ro[i + 2], r + dx / 2.0);
        pR = linear(r, p1, r + dx, p3, r + 2.0 * dx, p[i + 2], r + dx / 2.0);
        ro_pR = linear(r, ro_p1, r + dx, ro_p3, r + 2.0 * dx, ro_p[i + 2], r + dx / 2.0);
        p_pR = linear(r, p_p1, r + dx, p_p3, r + 2.0 * dx, p_p[i + 2], r + dx / 2.0);
        uR = linear(r, u1, r + dx, u3, r + 2.0 * dx, u[i + 2], r + dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro3;
        }
        if (pR <= 0.0)
        {
            pR = p3;
        }
        if (ro_pR <= 0.0)
        {
            ro_pR = ro_p3;
        }
        if (p_pR <= 0.0)
        {
            p_pR = p_p3;
        }

    }

    double PP, RO;
    //time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, ro_pL, pL, uL, 0.0, p_pL, roR, ro_pR, pR, uR, 0.0, p_pR, 0.0, P, PQ, 1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(roL, 1.0, pL, uL, 0.0, 0.0, br(r + dx / 2.0), by[i], 0.0, //
        roR, 1.0, pR, uR, 0.0, 0.0, br(r + dx / 2.0), by[i + 1], 0.0, P, PQ, 1.0, 0.0, 0.0, dx, 2));
    B1 = P[0];
    B2 = P[1];
    B3 = P[7];  // 4
    B4 = P[5];

    //time2 = min(time2, HLLC_2d_Korolkov_b_s(ro_pL, 1.0, p_pL, uL, 0.0, 1.0, ro_pR, 1.0, p_pR, uR, 0.0, 1.0, 0.0, P, PQ, 1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(ro_pL, 1.0, p_pL, uL, 0.0, 0.0, br(r + dx / 2.0), by[i], 0.0, //
        ro_pR, 1.0, p_pR, uR, 0.0, 0.0, br(r + dx / 2.0), by[i + 1], 0.0, P, PQ, 1.0, 0.0, 0.0, dx, 2));
    Bp1 = P[0];
    Bp2 = P[1];
    Bp3 = P[7];  // 4
    Bp4 = P[5];


    roL = ro1;
    pL = p1;
    ro_pL = ro_p1;
    p_pL = p_p1;
    uL = u1;
    roR = ro4;
    pR = p4;
    ro_pR = ro_p4;
    p_pR = p_p4;
    uR = u4;

    if (i > 1 && i < *N - 2)
    {
        roL = linear(r - dx, ro4, r, ro1, r + dx, ro3, r - dx / 2.0);
        pL = linear(r - dx, p4, r, p1, r + dx, p3, r - dx / 2.0);
        ro_pL = linear(r - dx, ro_p4, r, ro_p1, r + dx, ro_p3, r - dx / 2.0);
        p_pL = linear(r - dx, p_p4, r, p_p1, r + dx, p_p3, r - dx / 2.0);
        uL = linear(r - dx, u4, r, u1, r + dx, u3, r - dx / 2.0);
        if (roL <= 0.0)
        {
            roL = ro1;
        }
        if (pL <= 0.0)
        {
            pL = p1;
        }
        if (ro_pL <= 0.0)
        {
            ro_pL = ro_p1;
        }
        if (p_pL <= 0.0)
        {
            p_pL = p_p1;
        }

        roR = linear(r - 2.0 * dx, ro[i - 2], r - dx, ro4, r, ro1, r - dx / 2.0);
        pR = linear(r - 2.0 * dx, p[i - 2], r - dx, p4, r, p1, r - dx / 2.0);
        ro_pR = linear(r - 2.0 * dx, ro_p[i - 2], r - dx, ro_p4, r, ro_p1, r - dx / 2.0);
        p_pR = linear(r - 2.0 * dx, p_p[i - 2], r - dx, p_p4, r, p_p1, r - dx / 2.0);
        uR = linear(r - 2.0 * dx, u[i - 2], r - dx, u4, r, u1, r - dx / 2.0);
        if (roR <= 0.0)
        {
            roR = ro4;
        }
        if (pR <= 0.0)
        {
            pR = p4;
        }
        if (ro_pR <= 0.0)
        {
            ro_pR = ro_p4;
        }
        if (p_pR <= 0.0)
        {
            p_pR = p_p4;
        }

    }

    PP = 0.0;
    RO = 0.0;
    //time2 = min(time2, HLLC_2d_Korolkov_b_s(roL, ro_pL, pL, uL, 0.0, p_pL, roR, ro_pR, pR, uR, 0.0, p_pR, 0.0, P, PQ, -1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(roL, 1.0, pL, uL, 0.0, 0.0, br(r - dx / 2.0), by[i], 0.0, //
        roR, 1.0, pR, uR, 0.0, 0.0, br(r - dx / 2.0), by[i - 1], 0.0, P, PQ, -1.0, 0.0, 0.0, dx, 2));

    B1 = B1 + P[0];
    B2 = B2 + P[1];
    B3 = B3 + P[7];  // 4
    B4 = B4 + P[5];  // 4

    //time2 = min(time2, HLLC_2d_Korolkov_b_s(ro_pL, 1.0, p_pL, uL, 0.0, 1.0, ro_pR, 1.0, p_pR, uR, 0.0, 1.0, 0.0, P, PQ, -1.0, 0.0, dx, RO, PP));
    time2 = min(time2, HLLDQ_Korolkov(ro_pL, 1.0, p_pL, uL, 0.0, 0.0, br(r - dx / 2.0), by[i], 0.0, //
        ro_pR, 1.0, p_pR, uR, 0.0, 0.0, br(r - dx / 2.0), by[i - 1], 0.0, P, PQ, -1.0, 0.0, 0.0, dx, 2));

    Bp1 = Bp1 + P[0];
    Bp2 = Bp2 + P[1];
    Bp3 = Bp3 + P[7];
    Bp4 = Bp4 + P[5];




    // ??????? ????????? 2-??????

    double pp = p_p1;
    double roo = ro_p1;

    double roH = n_H * exp(-r0 / r);
    double UH = sqrt(kv(v_H - u1) + 4.0 / pi * (kv(c_H) + pp / roo));
    double UMH = sqrt(kv(v_H - u1) + 64.0 / (9.0 * pi) * (kv(c_H) + pp / roo));
    double nu = Kn * roo * roH * UMH * sigma(UMH);
    double Q1, Q2, Q3, Q1pi, Q3p;
    Q1 = 0.0;
    Q2 = nu * (v_H - u1);
    Q3 = nu * (0.5 * (kv(v_H) - kv(u1)) + (kv(c_H) - pp / roo) * UH / UMH);
    Q1pi = -Kn * roo * roH * UH * sigma(UH);// +0.0 * nu * (0.5 * (kv(v_H - u1)) + (kv(c_H) - p_pp / ro_pp) * UH / UMH);
    Q3p = 0.0; //0.03 * nu * (0.5 * (kv(v_H - u1)) + (kv(c_H) - pp / roo) * UH / UMH);

    //printf("%lf, %lf\n", Q3, Q3p);

    double bb = br(r) * br(r) + by[i] * by[i];
    double sk = br(r) * u1;

    //double ddb = ((br(r) * br(r) + by[i] * by[i] / 2.0) - (br(r - dx) * br(r - dx) + by[i - 1] * by[i - 1] / 2.0)) / (4.0 * pi * dx);

    ro2[i] = -*T_do * (B1 / dx - Q1 - (2.0 / r) * (-ro1 * u1)) + ro1;

    if (ro2[i] < 0.0)
    {
        printf("ERROR  707  ro < 0 = %lf __  %lf \n", r, ro2[i]);
        ro2[i] = 0.000001;
    }

    u2[i] = (-*T_do * (B2 / dx - Q2 + (2.0 / r) * (ro1 * u1 * u1 - br(r) * br(r)/(4.0 * pi) + by[i] * by[i]/(8.0 * pi))) + ro1 * u1) / ro2[i];
    //u2[i] = (-*T_do * (B2 / dx + ddb - Q2 + (2.0 / r) * (ro1 * u1 * u1 + (bb + by[i] * by[i])/(8.0 * pi))) + ro1 * u1) / ro2[i];

    by2[i] = -*T_do * (B4 / dx + (1.0 / r) * (by[i] * u1)) + by[i];

    double bb2 = br(r) * br(r) + by2[i] * by2[i];

    p2[i] = (-*T_do * (B3 / dx - Q3 + (2.0 / r) * ((ggg * p1 * u1) / (ggg - 1.0) + ro1 * u1 * u1 * u1 / 2.0 + u1 * bb / (4.0 * pi) - br(r)*sk / (4.0 * pi))) + //
        p1 / (ggg - 1.0) + (ro1 * u1 * u1 / 2.0 - ro2[i] * u2[i] * u2[i] / 2.0 ) + (bb/(8.0 * pi)  - bb2/(8.0 * pi))) * (ggg - 1.0);

    //p2[i] = (-*T_do * (B3 / dx - Q3 + (2.0 / r) * ((ggg * p1 * u1) / (ggg - 1.0) + ro1 * u1 * u1 * u1 / 2.0 + u1 * by[i] * by[i] /(4.0 * pi))) + //
    //    p1 / (ggg - 1.0) + (ro1 * u1 * u1 / 2.0 - ro2[i] * u2[i] * u2[i] / 2.0) + (bb / (8.0 * pi) - bb2 / (8.0 * pi))) * (ggg - 1.0);
   
    if (p2[i] < 0.0)
    {
        printf("ERROR  700  p2 < 0 = %lf __  %10lf \n", r, p2[i]);
        printf("%13.10lf __  %13.10lf__  %10lf__  %10lf \n", p1 / (ggg - 1.0), (ro1* u1* u1 / 2.0 - ro2[i] * u2[i] * u2[i] / 2.0), //
            (bb / (8.0 * pi) - bb2 / (8.0 * pi)), p1 / (ggg - 1.0) + (ro1 * u1 * u1 / 2.0 - ro2[i] * u2[i] * u2[i] / 2.0) + (bb / (8.0 * pi) - bb2 / (8.0 * pi)));
        p2[i] = 0.000001;
    }

    ro_p2[i] =  -*T_do * (Bp1 / dx - Q1pi - (2.0 / r) * (-ro_p1 * u1)) + ro_p1;
    double Q = (ro_p2[i] * u2[i] - ro_p1 * u1) / *T_do + Bp2 / dx + (2.0 / r) * (ro_p1 * u1 * u1 - br(r) * br(r) / (4.0 * pi) + by[i] * by[i] / (8.0 * pi));
    
    //p_p2[i] = ((-*T_do * (Bp3 / dx - (Q3p - Q1pi * (0.5 * kv(u1) + p_p1 / g1) + Q * u1) - (2.0 / r) * (-(ggg * p_p1 * u1) / (ggg - 1.0) - ro_p1 * u1 * u1 * u1 / 2.0 - u1 * bb / (4.0 * pi) + br(r) * sk / (4.0 * pi) - u1 * bb / (4.0 * pi) + br(r) * sk / (4.0 * pi))) + //
    //    p_p1 / (ggg - 1.0) + ro_p1 * u1 * u1 / 2.0 + bb / (8.0 * pi)) - ro_p2[i] * u2[i] * u2[i] / 2.0 - bb2 / (8.0 * pi)) * (ggg - 1.0);

    p_p2[i] = (-*T_do * (Bp3 / dx - (Q3p - Q1pi * (0.5 * kv(u1) + p_p1 / g1) + Q * u1) + (2.0 / r) * ((ggg * p_p1 * u1) / (ggg - 1.0) + ro_p1 * u1 * u1 * u1 / 2.0 + u1 * bb / (4.0 * pi) - br(r) * sk / (4.0 * pi))) + //
        p_p1 / (ggg - 1.0) + (ro_p1 * u1 * u1 / 2.0 - ro_p2[i] * u2[i] * u2[i] / 2.0 ) + (bb / (8.0 * pi)  - bb2 / (8.0 * pi)) ) * (ggg - 1.0);



    if (p_p2[i] < 0.0)
    {
        printf("ERROR  707  pp2 < 0 = %lf __  %13.10lf  __  %13.10lf \n", r, p_p2[i], p2[i]);
        printf("%13.10lf __  %13.10lf__  %13.10lf__  %13.10lf __ \n %13.10lf__  %13.10lf\n", p_p1 / (ggg - 1.0), (ro_p1* u1* u1 / 2.0 - ro_p2[i] * u2[i] * u2[i] / 2.0), //
            (bb / (8.0 * pi) - bb2 / (8.0 * pi)), p_p1 / (ggg - 1.0) + (ro_p1 * u1 * u1 / 2.0 - ro_p2[i] * u2[i] * u2[i] / 2.0) + (bb / (8.0 * pi) - bb2 / (8.0 * pi)), //
            Q, (Bp3 / dx - (Q3p - Q1pi * (0.5 * kv(u1) + p_p1 / g1) + Q * u1) - (2.0 / r) * (-(ggg * p_p1 * u1) / (ggg - 1.0) - ro_p1 * u1 * u1 * u1 / 2.0 - u1 * bb / (4.0 * pi) + br(r) * sk / (4.0 * pi) - u1 * bb / (4.0 * pi) + br(r) * sk / (4.0 * pi))));
        p_p2[i] = 0.000001;
    }

    if (ro_p2[i] < 0.0)
    {
        printf("ERROR  707  ro2 < 0 = %lf __  %lf \n", r, ro_p2[i]);
        ro_p2[i] = ro2[i];
    }

    if (time2 < *T)
    {
        *T = time2;
    }

}



int main()
{
    double* ro, * p, * u;

    double* swap_;
    int N = 8800;



    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(ro, p, u, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double *ro, double* p, double* u, int& N)
{
    cout << "Start -2" << endl;

    double * v, * w;
    double* ro2, * p2;
    double* dev_ro, * dev_p, * dev_u, * dev_v, * dev_w, * dev_ro2, * dev_p2, * dev_u2, * dev_v2, * dev_w2, * dev_ro_p, * dev_ro_p2, * dev_p_p, * dev_p_p2;
    double* dev_by, * dev_by2, *by;
    double* ro_, * p_, * u_, * v_, * w_, * rop_, * pp_, *by_;
    double* dev_T_all;
    double* dev_T_do;
    double* T_all;
    double* T_do, * T, * dev_T;
    double* Time_Omni, * Ro_Omni, * P_Omni, * U_Omni, * V_Omni, * W_Omni;
    double* Time_V, * Ro_V, * P_V, * U_V, * Dist_V;
    double* dev_Time_Omni, * dev_Ro_Omni, * dev_P_Omni, * dev_U_Omni, * dev_V_Omni, * dev_W_Omni;
    double* dev_Time_V, * dev_Ro_V, * dev_P_V, * dev_U_V, * dev_Dist_V;
    cudaError_t cudaStatus;
    int N_V = data_voy;               // ????? ?????? ????????, ?????? ???? ?????????????
    string s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    int a1, a2, a3, a4;
    double b1, b2, b3, b4, b5, b6, b7, b8, b9, b10;
    int N_O1 = Omni_, N_O2 = 0;     // ?????? ????-??????? ?????? ???? ?????????? (?????? ??? ?? ???? ??? ???????? ?? 100.000)
    int* dev_mas_V, * mas_V;
    int* dev_mas_Omni, * mas_Omni;
    int* dev_N;
    int* NN;
    bool* dev_boo;
    double* j1, * j2, * j3, * dev_j1, * dev_j2, * dev_j3;
    ofstream fout2;
    int step = 0;
    int* voy, * dev_voy;  // ???????? ?? ?????? ?????????
    bool* boo;
    double kk = 1.0;
    int kk_int = 0;
    ofstream fout5;
    fout5.open("voyadger2_calculations.txt");// , ios_base::out | ios_base::app);

    cout << "Start -1" << endl;

    boo = new bool[1];
    NN = new int[1];
    *NN = N;
    Time_V = new double[N_V];
    Dist_V = new double[N_V];
    U_V = new double[N_V];
    Ro_V = new double[N_V];
    P_V = new double[N_V];
    T_all = new double[1];
    Ro_Omni = new double[N_O1];
    P_Omni = new double[N_O1];
    U_Omni = new double[N_O1];
    V_Omni = new double[N_O1];
    W_Omni = new double[N_O1];
    Time_Omni = new double[N_O1];
    mas_Omni = new int[1];
    ro = new double[N];
    p = new double[N];
    u = new double[N];
    v = new double[N];
    w = new double[N];
    ro2 = new double[N];
    p2 = new double[N];
    ro_ = new double[N];
    p_ = new double[N];
    rop_ = new double[N];
    pp_ = new double[N];
    u_ = new double[N];
    v_ = new double[N];
    w_ = new double[N];
    by = new double[N];
    by_ = new double[N];
    voy = new int[1];
    mas_V = new int[1];
    j1 = new double[1];
    j2 = new double[1];
    j3 = new double[1];

    *voy = 0;
    *mas_V = 0;

    *T_all = 100.0; // 3600.01 * minut; //  50.0; // 3600.01 * minut;

    cout << "Start 0" << endl;

    //takeDataOmni(Ro_Omni, P_Omni, U_Omni, Time_Omni, N_O1, N_O2);
    takeDataOmni2(Ro_Omni, P_Omni, U_Omni, V_Omni, W_Omni, Time_Omni, N_O1, N_O2);

    cout << "Start 01" << endl;
    // ????????? ????????? ???????

    for (int i = 0; i < N; i++)
    {
        double r = L + i * dx;
        double pE = 0.0059994;
        ro[i] = 1.0 / (r * r);
        ro2[i] = 1.0 / (r * r); //0.000001 / (r * r);
        p[i] =  pE * pow(1.0 / r, 2.0 * ggg);
        u[i] = 1.0;
        v[i] = 0.0;
        w[i] = 0.0;
        by[i] = 0.292303/r;
    }

    cout << "Start 1" << endl;

    // ????????? ?????? ????????
    if (true)
    {
        ifstream fout4;
        fout4.open("voyager2_all_data_mhd.txt");
        fout4 >> s1 >> s2 >> s3 >> s4 >> s5 >> s6 >> s7 >> s8 >> s9 >> s10 >> s11;
        for (int i = 0; i < N_V; i++)
        {
            fout4 >> a1;
            fout4 >> b1 >> b2 >> b3 >> b4 >> b5 >> b6 >> b7 >> b8 >> b9 >> b10;
            Time_V[i] = a1 * hour;
            Dist_V[i] = b1;
            U_V[i] = b7;
            Ro_V[i] = b6;
            P_V[i] = b5;
        }
        fout4.close();
    }

    cout << "Start 2" << endl;



    // ???????? ?????? ?? CUDA
    if (true)
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_ro, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_u, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_v, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_w, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_by, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_by2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_ro2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_u2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }  

        cudaStatus = cudaMalloc((void**)&dev_v2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_w2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_Time_V, N_V * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_Dist_V, N_V * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_Time_Omni, N_O1 * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_Ro_Omni, N_O1 * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_P_Omni, N_O1 * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_U_Omni, N_O1 * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_V_Omni, N_O1 * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_W_Omni, N_O1 * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_T_all, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_T, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_T_do, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_mas_V, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_N, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_mas_Omni, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_mas_V, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_voy, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        } 

        cudaStatus = cudaMalloc((void**)&dev_boo, sizeof(bool));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_j1, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_j2, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_j3, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_ro_p, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p_p, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_ro_p2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p_p2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
    }

    cout << "Start 3" << endl;

    // ???????? ?????? ?? CUDA
    if (true)
    {
        cudaStatus = cudaMemcpy(dev_Time_V, Time_V, N_V * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_Dist_V, Dist_V, N_V * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_Time_Omni, Time_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_U_Omni, U_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_V_Omni, V_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_W_Omni, W_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_Ro_Omni, Ro_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_P_Omni, P_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ro, ro, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ro2, ro, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_by, by, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_by2, by, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_p, p, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_p2, p, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_u, u, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_v, v, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_w, w, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_u2, u, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_v2, v, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_w2, w, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_T_all, T_all, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_N, NN, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_mas_V, mas_V, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_voy, voy, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ro_p, ro2, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_ro_p2, ro2, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_p_p, p, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_p_p2, p, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }

    cout << "Start 4" << endl;

    init_time << <1, 1 >> > (dev_T_do, dev_T);   // ????????????? ?????? ??? ?? ???????

    

    while (*mas_V < 127470) // (*T_all < 60.0)//
    {
        step++;
        //takeOmni << <1, 1 >> > (dev_ro, dev_p, dev_ro_p, dev_p_p, dev_u, dev_Ro_Omni, dev_P_Omni, dev_U_Omni, dev_Time_Omni, dev_T_all, dev_mas_Omni);
        takeOmni2 << <1, 1 >> > (dev_ro, dev_p, dev_ro_p, dev_p_p, dev_u, dev_v, dev_w, dev_Ro_Omni, dev_P_Omni, dev_U_Omni, dev_V_Omni, dev_W_Omni, dev_Time_Omni, dev_T_all, dev_mas_Omni);
       
        takeVoyadger << <1, 1 >> > (dev_T_do, dev_T_all, dev_Time_V, dev_mas_V, dev_voy);

        // Launch a kernel on the GPU with one thread for each element.
        //calcul << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro, dev_u, dev_p, dev_ro2, dev_u2, dev_p2,//
        //    dev_T, dev_T_do, dev_N);

        //calcul_component << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro, dev_u, dev_p, dev_ro2, dev_u2, dev_p2,//
        //    dev_ro_p, dev_ro_p2, dev_p_p, dev_p_p2, dev_T, dev_T_do, dev_N);

        do
        {
            if (*mas_V > kk_int + 100)
            {
                kk = 1.0;
            }

            *boo = true;
            cudaStatus = cudaMemcpy(dev_boo, boo, sizeof(bool), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }
            
            calcul_component_2 << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro, dev_u, dev_v, dev_w, dev_p, dev_ro2, dev_u2, dev_v2, dev_w2, dev_p2,//
                dev_ro_p, dev_ro_p2, dev_p_p, dev_p_p2, dev_T, dev_T_do, dev_N, dev_mas_V, dev_boo, kk);
            
            cudaStatus = cudaMemcpy(boo, dev_boo, sizeof(bool), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            if (*boo == false)
            {
                kk = kk * 1.3;
                kk_int = *mas_V;
                funk_time_devide << <1, 1 >> > (dev_T_do);
            }
        } while (*boo == false);

        //calcul_component_mgd << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro, dev_u, dev_p, dev_ro2, dev_u2, dev_p2,//
        //    dev_ro_p, dev_ro_p2, dev_p_p, dev_p_p2, dev_by, dev_by2, dev_T, dev_T_do, dev_N);

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_T_all);

        cudaStatus = cudaMemcpy(voy, dev_voy, sizeof(int), cudaMemcpyDeviceToHost);
        
        // ????? ?????????? ?????? ????????
        if (*voy == 1)
        {
            *voy == 0;
            cudaStatus = cudaMemcpy(dev_voy, voy, sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 5!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(mas_V, dev_mas_V, sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 6!");
                goto Error;
            }

            //takeVoyadger2 << <1, 1 >> > (dev_ro2, dev_ro_p2, dev_p_p2, dev_u2, dev_Dist_V, dev_mas_V, dev_j1, dev_j2, dev_j3);
            takeVoyadger2_2 << <1, 1 >> > (dev_ro2, dev_ro_p2, dev_p_p2, dev_u2, dev_v2, dev_w2, dev_Dist_V, dev_mas_V, dev_j1, dev_j2, dev_j3);
            cudaStatus = cudaMemcpy(j1, dev_j1, sizeof(double), cudaMemcpyDeviceToHost);
            cudaStatus = cudaMemcpy(j2, dev_j2, sizeof(double), cudaMemcpyDeviceToHost);
            cudaStatus = cudaMemcpy(j3, dev_j3, sizeof(double), cudaMemcpyDeviceToHost);
            cudaStatus = cudaMemcpy(T_all, dev_T_all, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }
            fout5 << *T_all << " " << Dist_V[*mas_V] << " " << *j1 * 7.0 << " " << *j2 * 375.0 << " " << *j3 / (2.0 * *j1 * 5.86922 * 0.00000001) << //
                " " << Ro_V[*mas_V] << " " << U_V[*mas_V] << " " << P_V[*mas_V] << endl;

            // ? ???? ?????? ????? ???????? ?????? ?????? ??? ???????? (????????? NN)
            // ????????? ? ???? ??????
            if (Dist_V[*mas_V] > * NN * dx - 8.0)
            {
                int kl = (int)((Dist_V[*mas_V] + 16.0) / dx);
                cout << "New size do " << kl * dx <<  "  " << kl << " " << *NN << endl;
               
                cudaStatus = cudaMemcpy(ro_, dev_ro2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }

                cudaStatus = cudaMemcpy(by_, dev_by2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
               
                cudaStatus = cudaMemcpy(p_, dev_p2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }

                cudaStatus = cudaMemcpy(rop_, dev_ro_p2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                
                cudaStatus = cudaMemcpy(pp_, dev_p_p2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
               
                cudaStatus = cudaMemcpy(u_, dev_u2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }

                cudaStatus = cudaMemcpy(v_, dev_v2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }

                cudaStatus = cudaMemcpy(w_, dev_w2, *NN * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaFree(dev_ro);
                cudaFree(dev_p);
                cudaFree(dev_u);
                cudaFree(dev_v);
                cudaFree(dev_w);
                cudaFree(dev_ro2);
                cudaFree(dev_p2);
                cudaFree(dev_u2);
                cudaFree(dev_v2);
                cudaFree(dev_w2);
                cudaFree(dev_ro_p);
                cudaFree(dev_p_p);
                cudaFree(dev_ro_p2);
                cudaFree(dev_p_p2);
                cudaFree(dev_by);
                cudaFree(dev_by2);
                delete[] ro;
                delete[] p;
                delete[] ro2;
                delete[] p2;
                delete[] u;
                delete[] v;
                delete[] w;
                delete[] by;
                ro = new double[kl];
                p = new double[kl];
                ro2 = new double[kl];
                p2 = new double[kl];
                u = new double[kl];
                v = new double[kl];
                w = new double[kl];
                by = new double[kl];
                cudaStatus = cudaMalloc((void**)&dev_ro, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_by, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_p, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_u, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_v, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_w, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_ro2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_by2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_p2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_u2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_v2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_w2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }

                cudaStatus = cudaMalloc((void**)&dev_ro_p, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_p_p, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_ro_p2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }
                cudaStatus = cudaMalloc((void**)&dev_p_p2, kl * sizeof(double));
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMalloc failed!");
                    goto Error;
                }

                for (int i = 0; i < *NN; i++)
                {
                    ro[i] = ro_[i];
                    p[i] = p_[i];
                    ro2[i] = rop_[i];
                    p2[i] = pp_[i];
                    u[i] = u_[i];
                    v[i] = v_[i];
                    w[i] = w_[i];
                    by[i] = by_[i];
                }
                for (int i = *NN; i < kl; i++)
                {
                    double r = L + i * dx;
                    double pE = 0.0059994;
                    ro[i] = 1.0 / (r * r);
                    ro2[i] = 1.0 / (r * r);  //0.000001 / (r * r);
                    p[i] = pE * pow(1.0 / r, 2.0 * ggg);
                    p2[i] = pE * pow(1.0 / r, 2.0 * ggg);
                    u[i] = 1.0;
                    v[i] = 0.0;
                    w[i] = 0.0;
                    by[i] = 0.292303 / r;
                }
                cudaStatus = cudaMemcpy(dev_ro2, ro, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_by2, by, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_p2, p, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_u2, u, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_v2, v, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_w2, w, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_ro_p2, ro2, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_p_p2, p2, kl * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                N = kl;
                *NN = kl;
                cudaStatus = cudaMemcpy(dev_N, NN, sizeof(int), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                delete[] ro_;
                delete[] p_;
                delete[] rop_;
                delete[] pp_;
                delete[] u_;
                delete[] v_;
                delete[] w_;
                delete[] by_;
                ro_ = new double[kl];
                p_ = new double[kl];
                u_ = new double[kl];
                v_ = new double[kl];
                w_ = new double[kl];
                rop_ = new double[kl];
                pp_ = new double[kl];
                by_ = new double[kl];
            }
        }

        //takeOmni << <1, 1 >> > (dev_ro2, dev_p2, dev_ro_p2, dev_p_p2, dev_u2, dev_Ro_Omni, dev_P_Omni, dev_U_Omni, dev_Time_Omni, dev_T_all, dev_mas_Omni);
        takeOmni2 << <1, 1 >> > (dev_ro2, dev_p2, dev_ro_p2, dev_p_p2, dev_u2, dev_v2, dev_w2, dev_Ro_Omni, dev_P_Omni, dev_U_Omni, dev_V_Omni, dev_W_Omni, dev_Time_Omni, dev_T_all, dev_mas_Omni);
        
        takeVoyadger << <1, 1 >> > (dev_T_do, dev_T_all, dev_Time_V, dev_mas_V, dev_voy);
        
        // Launch a kernel on the GPU with one thread for each element.
        //calcul << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro2, dev_u2, dev_p2, dev_ro, dev_u, dev_p,//
        //    dev_T, dev_T_do, dev_N);

        //calcul_component << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro2, dev_u2, dev_p2, dev_ro, dev_u, dev_p,//
        //    dev_ro_p2, dev_ro_p, dev_p_p2, dev_p_p, dev_T, dev_T_do, dev_N);


        do
        {
            if (*mas_V > kk_int + 100)
            {
                kk = 1.0;
            }

            *boo = true;
            cudaStatus = cudaMemcpy(dev_boo, boo, sizeof(bool), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            calcul_component_2 << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro2, dev_u2, dev_v2, dev_w2, dev_p2, dev_ro, dev_u, dev_v, dev_w, dev_p,//
                dev_ro_p2, dev_ro_p, dev_p_p2, dev_p_p, dev_T, dev_T_do, dev_N, dev_mas_V, dev_boo, kk);

            cudaStatus = cudaMemcpy(boo, dev_boo, sizeof(bool), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            if (*boo == false)
            {
                kk = kk * 1.3;
                kk_int = *mas_V;
                funk_time_devide << <1, 1 >> > (dev_T_do);
            }
        } while (*boo == false);

        //calcul_component_mgd << < (int)(1.0 * *NN / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (dev_ro2, dev_u2, dev_p2, dev_ro, dev_u, dev_p,//
        //    dev_ro_p2, dev_ro_p, dev_p_p2, dev_p_p, dev_by2, dev_by, dev_T, dev_T_do, dev_N);

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_T_all);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(voy, dev_voy, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // ????? ?????????? ?????? ????????
        if (*voy == 1)
        {
            *voy == 0;
            cudaStatus = cudaMemcpy(dev_voy, voy, sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }


            cudaStatus = cudaMemcpy(mas_V, dev_mas_V, sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            takeVoyadger2 << <1, 1 >> > (dev_ro, dev_ro_p, dev_p_p, dev_u, dev_Dist_V, dev_mas_V, dev_j1, dev_j2, dev_j3); // ???? 2 ?????-??
            takeVoyadger2_2 << <1, 1 >> > (dev_ro, dev_ro_p, dev_p_p, dev_u, dev_v, dev_w, dev_Dist_V, dev_mas_V, dev_j1, dev_j2, dev_j3); // ???? 2 ?????-??
            cudaStatus = cudaMemcpy(j1, dev_j1, sizeof(double), cudaMemcpyDeviceToHost);
            cudaStatus = cudaMemcpy(j2, dev_j2, sizeof(double), cudaMemcpyDeviceToHost);
            cudaStatus = cudaMemcpy(j3, dev_j3, sizeof(double), cudaMemcpyDeviceToHost);
            cudaStatus = cudaMemcpy(T_all, dev_T_all, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 7!");
                goto Error;
            }
            fout5 << *T_all << " " << Dist_V[*mas_V] << " " << *j1 * 7.0 << " " << *j2 * 375.0 << " " << *j3 / (2.0 * *j1 * 5.86922 * 0.00000001) << //
                " " << Ro_V[*mas_V] << " " << U_V[*mas_V] << " " << P_V[*mas_V] << endl;
        }

        if (step % 10000 == 0)
        {
            cudaStatus = cudaMemcpy(T_all, dev_T_all, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 8!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(mas_Omni, dev_mas_Omni, sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 9!");
                goto Error;
            }
            cout << *T_all << " " << *mas_Omni << " " << *mas_V << " " << kk << endl;

            // ????? ????????? ?????????? ???????? ????
            if (*mas_Omni > 0.85 * N_O1)
            {
                N_O2 += (*mas_Omni - 1);
                //takeDataOmni(Ro_Omni, P_Omni, U_Omni, Time_Omni, N_O1, N_O2);
                takeDataOmni2(Ro_Omni, P_Omni, U_Omni, V_Omni, W_Omni, Time_Omni, N_O1, N_O2);
                cudaStatus = cudaMemcpy(dev_Time_Omni, Time_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed 10!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_U_Omni, U_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed 11!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_V_Omni, V_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed 11!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_W_Omni, W_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed 11!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_Ro_Omni, Ro_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed 12!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(dev_P_Omni, P_Omni, N_O1 * sizeof(double), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                *mas_Omni = 0;
                cudaStatus = cudaMemcpy(dev_mas_Omni, mas_Omni, sizeof(int), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!");
                    goto Error;
                }
                cout << "Refrishing the Omny's arrays" << endl;
            }
        }


        // ?????? ???????? 
        if (step % 1000000 == 0)
        {
            cudaStatus = cudaMemcpy(T_all, dev_T_all, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 8!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(ro, dev_ro, *NN * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(p, dev_p, *NN * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(ro2, dev_ro_p, *NN * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(p2, dev_p_p, *NN * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(u, dev_u, *NN * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                goto Error;
            }

            fout2.open("save.txt");
            fout2 << *T_all << " " << *NN << endl;
            for (int i = 0; i < *NN; i++)
            {
                double r = L + i * dx;
                fout2 << r << " " << ro[i] << " " << p[i] << " " << u[i] << " " << ro2[i] << " " << p2[i] << endl;
                //fout2 << r << " " << (ro[i] - ro2[i]) * 7.0 << " " << p[i] << " " << u[i] * 375.0 << " " << p[i] / (2.0 * (ro[i] - ro2[i]) * 5.86922 * 0.00000001) << endl;
            }
            fout2.close();

        }
    }


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ro, dev_ro, *NN * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(p, dev_p, *NN * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(ro2, dev_ro_p, *NN * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(p2, dev_p_p, *NN * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(u, dev_u, *NN * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(by, dev_by, *NN * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    
    fout2.open("save_2.txt");
    fout2 << *T_all << " " << *NN << endl;
    for (int i = 0; i < *NN; i++)
    {
        double r = L + i * dx;
        fout2 << r << " " << ro[i] << " " << p[i] << " " << u[i] << " " << ro2[i] << " " << p2[i] << " " << by[i] << endl;
        //fout2 << r << " " << (ro2[i] - ro[i])  * 7.0 << " " << p[i] << " " << u[i] * 375.0 << " " << p[i]/ (2.0 * (ro2[i] - ro[i]) * 5.86922 * 0.00000001) << " " << p2[i] / (2.0 * ro2[i] * 5.86922 * 0.00000001) <<  endl;
    }
    fout2.close();

Error:
    
    return cudaStatus;
}
