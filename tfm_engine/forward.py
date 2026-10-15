# -*- coding: utf-8 -*-
"""
tfm_engine.forward
==================
WLF 体系 1D TEM 正演引擎（层状介质 + 可选 IP 极化率）
完全遵循 WLF 设计原理:
  - 核心: Boerner/WLF Mod1 扩散深度等效加权 (与 v1 inversion.forward_rhoa 接口完全对齐,
          因此正演↔反演↔预处理 Nabighian ρa 公式三向天生量纲自洽, 无缩放bug)
  - 真频域正演: 含空气界面反射系数递推 + Guptasarma J1 滤波 (中心回线核)
                + Key(2012) 201点正弦DLF 频→时转换 → tem_forward_1d_freq
  - IP: Pelton Cole-Cole 时域近似乘子 (与 WLF pcvogoss_pc.m 的 IP 正演支路一致)
  - 观测方式: 中心回线 VMD dBz/dt (输出 Vt 的量纲与 preprocess.py 提取的 Hx_gated 对齐,
          因此 apparent_resistivity_dbzdt 的 RHOA_SCALE_CAL=1e6 可直接复用)

冻结函数签名 (GUI / 反演模块统一依赖, 禁止修改形参):
    model = LayeredModel(thickness_m, rho_ohm_m, chargeability=None, tau_s=None, c_exp=None)
    (t_sec, dbzdt) = tem_forward_1d(model, tx_area_m2, tx_current_a, gates_sec, include_ip=True)
    rhoa_vals       = apparent_resistivity_dbzdt(dbzdt, t_sec, tx_area_m2, tx_current_a)
100% 离线: 无外部依赖, 纯 numpy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

# ============================================================
# 物理 + 校准 全局常量 (与 preprocess.py / inversion.py 完全统一)
# ============================================================
_MU0 = 4 * np.pi * 1e-7              # H/m (真空磁导率, SI)
_RHOA_SCALE_CAL_DEFAULT = 1_000_000  # 与 preprocess.DEFAULT_CFG['RHOA_SCALE_CAL'] 保持 1e6 (统一!)
_Z_FINE_MAX_M = 10_000.0             # 扩散深度加权细网格上限 (10km, 足够覆盖 100Ω·m×20ms)
_Z_N_FINE = 600                      # 每正演调用 600 深度点 (精度 ~ 1.5% 且 <1ms 计算)

# ============================================================
# Guptasarma & Singh (1997) 汉克尔数字线性滤波系数
# Geophysical Prospecting 45: 745-762; 数值取自 libdlf 权威表并内嵌, 保证 100% 离线
#   J0: 61 点 (gupt_61_1997);  J1: 47 点 (gupt_47_1997, 中心回线核使用)
# 约定: ∫ f(λ) Jν(λr) dλ ≈ (1/r) Σ w_i f(base_i / r)
# ============================================================
_GUPT61_BASE = np.array([
    8.26989508568e-06, 1.08177839572e-05, 1.41506571161e-05, 1.85103619753e-05, 2.42132572110e-05, 3.16731690903e-05,
    4.14314204602e-05, 5.41961114299e-05, 7.08935021172e-05, 9.27352260125e-05, 1.21306211243e-04, 1.58679689681e-04,
    2.07567639440e-04, 2.71517577514e-04, 3.55169982652e-04, 4.64595028183e-04, 6.07733059534e-04, 7.94970779381e-04,
    1.03989495084e-03, 1.36027831064e-03, 1.77936923427e-03, 2.32757873672e-03, 3.04468722472e-03, 3.98273113176e-03,
    5.20977890244e-03, 6.81487032753e-03, 8.91447764880e-03, 1.16609572790e-02, 1.52536054291e-02, 1.99531198873e-02,
    2.61005173555e-02, 3.41418790682e-02, 4.46607203386e-02, 5.84203329048e-02, 7.64191726160e-02, 9.99633116236e-02,
    1.30761212516e-01, 1.71047701612e-01, 2.23746137434e-01, 2.92680542006e-01, 3.82853088108e-01, 5.00807078151e-01,
    6.55101753955e-01, 8.56933391635e-01, 1.12094775089e+00, 1.46630283344e+00, 1.91805906889e+00, 2.50899780582e+00,
    3.28200006544e+00, 4.29315817038e+00, 5.61584604156e+00, 7.34604352108e+00, 9.60930107669e+00, 1.25698502762e+01,
    1.64425211267e+01, 2.15083310509e+01, 2.81348767037e+01, 3.68030083439e+01, 4.81417223692e+01, 6.29738039624e+01,
    8.23755318740e+01])
_GUPT61_J0 = np.array([
    3.30220475766e-04,-1.18223623458e-03, 2.01879495264e-03,-2.13218719891e-03, 1.60839063172e-03,-9.09156346708e-04,
    4.37889252738e-04,-1.55298878782e-04, 7.98411962729e-05, 4.37268394072e-06, 3.94253441247e-05, 4.02675924344e-05,
    5.66053344653e-05, 7.25774926389e-05, 9.55412535465e-05, 1.24699163157e-04, 1.63262166579e-04, 2.13477133718e-04,
    2.79304232173e-04, 3.65312787897e-04, 4.77899413107e-04, 6.25100170825e-04, 8.17726956451e-04, 1.06961339341e-03,
    1.39920928148e-03, 1.83020380399e-03, 2.39417015791e-03, 3.13158560774e-03, 4.09654426763e-03, 5.35807925630e-03,
    7.00889482693e-03, 9.16637526490e-03, 1.19891721272e-02, 1.56755740646e-02, 2.04953856060e-02, 2.67778388247e-02,
    3.49719672729e-02, 4.55975312615e-02, 5.93498881451e-02, 7.69179091244e-02, 9.91094769804e-02, 1.26166963993e-01,
    1.57616825575e-01, 1.89707800260e-01, 2.13804195282e-01, 2.08669340316e-01, 1.40250562745e-01,-3.65385242807e-02,
   -2.98004010732e-01,-4.21898149249e-01, 5.94373771266e-02, 5.29621428353e-01,-4.41362405166e-01, 1.90355040550e-01,
   -6.19966386785e-02, 1.87255115744e-02,-5.68736766738e-03, 1.68263510609e-03,-4.38587145792e-04, 8.59117336292e-05,
   -9.15853765160e-06])
_GUPT47_BASE = np.array([
    8.89647830567e-04, 1.14767027689e-03, 1.48052636020e-03, 1.90991990243e-03, 2.46384943338e-03, 3.17843382995e-03,
    4.10026744107e-03, 5.28945826395e-03, 6.82354727545e-03, 8.80256447765e-03, 1.13555513365e-02, 1.46489749076e-02,
    1.88975822911e-02, 2.43784031786e-02, 3.14488135246e-02, 4.05698381826e-02, 5.23362119487e-02, 6.75151591391e-02,
    8.70964203148e-02, 1.12356788140e-01, 1.44943360423e-01, 1.86980938832e-01, 2.41210576218e-01, 3.11168306475e-01,
    4.01415711004e-01, 5.17837355824e-01, 6.68024493649e-01, 8.61770050184e-01, 1.11170717010e+00, 1.43413295901e+00,
    1.85007113333e+00, 2.38664286800e+00, 3.07883522787e+00, 3.97178249308e+00, 5.12370913181e+00, 6.60972631637e+00,
    8.52672953389e+00, 1.09997166394e+01, 1.41899383188e+01, 1.83054124114e+01, 2.36144876757e+01, 3.04633414237e+01,
    3.92985519498e+01, 5.06962175907e+01, 6.53995211143e+01, 8.43671888209e+01, 1.08836004122e+02])
_GUPT47_J1 = np.array([
    3.17926147465e-06,-9.73811660718e-06, 1.64866227408e-05,-1.81501261160e-05, 1.87556556369e-05,-1.46550406038e-05,
    1.53799733803e-05,-6.95628273934e-06, 1.41881555665e-05, 3.41445665537e-06, 2.13941715512e-05, 2.34962369042e-05,
    4.84340283290e-05, 7.33732978590e-05, 1.27703784430e-04, 2.08120025730e-04, 3.49803898913e-04, 5.79107814687e-04,
    9.65887918451e-04, 1.60401273703e-03, 2.66903777685e-03, 4.43111590040e-03, 7.35631696247e-03, 1.21782796293e-02,
    2.01097829218e-02, 3.30096953061e-02, 5.37143591532e-02, 8.60516613299e-02, 1.34267607144e-01, 2.00125033067e-01,
    2.74027505792e-01, 3.18168749246e-01, 2.41655667461e-01,-5.40549161658e-02,-4.46912952135e-01,-1.92231885629e-01,
    5.52376753950e-01,-3.57429049025e-01, 1.41510519002e-01,-4.61421935309e-02, 1.48273761923e-02,-5.07479209193e-03,
    1.83829713749e-03,-6.67742804324e-04, 2.21277518118e-04,-5.66248732755e-05, 7.88229202853e-06])

# ============================================================
# Key (2012) 频率→时间 正弦数字滤波系数 (201点, key_201_2012)
# Geophysical Journal International 191: 1229-1237; 数值取自 libdlf 权威表并内嵌
# 约定: ∫ f(ω) sin(ωt) dω ≈ (1/t) Σ wsin_i f(base_i / t)
# ============================================================
_KEY201_BASE = np.array([
    9.1898135789795544e-07, 1.0560236258145801e-06, 1.2135021985967225e-06, 1.3944646218148563e-06, 1.6024129035298632e-06, 1.8413712856029165e-06,
    2.1159641213409353e-06, 2.4315053665758195e-06, 2.7941014160203776e-06, 3.2107692750032131e-06, 3.6895723534572921e-06, 4.2397765100647990e-06,
    4.8720293663446341e-06, 5.5985663607919089e-06, 6.4334475306554683e-06, 7.3928296035848751e-06, 8.4952786646994725e-06, 9.7621294498526205e-06,
    1.1217898218180574e-05, 1.2890757193898812e-05, 1.4813079758803961e-05, 1.7022066946117209e-05, 1.9560467359658073e-05, 2.2477404450316962e-05,
    2.5829327159387539e-05, 2.9681102325729859e-05, 3.4107269997169212e-05, 3.9193485939077079e-05, 4.5038179255745865e-05, 5.1754457203059970e-05,
    5.9472294054641452e-05, 6.8341046380689574e-05, 7.8532343415514477e-05, 9.0243408448527221e-05, 1.0370087551467150e-04, 1.1916517524538292e-04,
    1.3693557475562244e-04, 1.5735596909953090e-04, 1.8082153637169376e-04, 2.0778638524439160e-04, 2.3877234293696414e-04, 2.7437905368319086e-04,
    3.1529558312353735e-04, 3.6231375319194681e-04, 4.1634346555562428e-04, 4.7843031014898881e-04, 5.4977579956440249e-04, 6.3176062087820603e-04,
    7.2597135488438745e-04, 8.3423117980991972e-04, 9.5863515369401899e-04, 1.1015907582204575e-03, 1.2658644886123726e-03, 1.4546353912032167e-03,
    1.6715565847497888e-03, 1.9208259560556996e-03, 2.2072673980160704e-03, 2.5364241622125167e-03, 2.9146661326298548e-03, 3.3493130964692705e-03,
    3.8487763976105395e-03, 4.4227217140193290e-03, 5.0822561091888648e-03, 5.8401429774594740e-03, 6.7110490428654943e-03, 7.7118281914628686e-03,
    8.8618476298971925e-03, 1.0183362682074693e-02, 1.1701947477049383e-02, 1.3446989862853833e-02, 1.5452260123909515e-02, 1.7756564507909106e-02,
    2.0404496209307014e-02, 2.3447298342546899e-02, 2.6943855605395223e-02, 3.0961833823176868e-02, 3.5578989426519304e-02, 4.0884674203786789e-02,
    4.6981564448367090e-02, 5.3987647963491189e-02, 6.2038507377358290e-02, 7.1289943955573878e-02, 8.1920992687256264e-02, 9.4137386993145250e-02,
    1.0817554010518914e-01, 1.2430712016577938e-01, 1.4284430758454120e-01, 1.6414583639372404e-01, 1.8862393651527715e-01, 2.1675231128725500e-01,
    2.4907530463166816e-01, 2.8621843526798946e-01, 3.2890050183176844e-01, 3.7794749315816500e-01, 4.3430857292399505e-01, 4.9907444798513595e-01,
    5.7349847587571534e-01, 6.5902091994412060e-01, 7.5729682151433553e-01, 8.7022802845825153e-01, 1.0000000000000000e+00, 1.1491241000036050e+00,
    1.3204861972090955e+00, 1.5174025129350848e+00, 1.7436837970197381e+00, 2.0037090739411751e+00, 2.3025103862617100e+00, 2.6458701753619409e+00,
    3.0404331839891712e+00, 3.4938350461726517e+00, 4.0148500529942019e+00, 4.6135609537963891e+00, 5.3015540788430497e+00, 6.0921435594709612e+00,
    7.0006289848698273e+00, 8.0445914816976902e+00, 9.2442339463025291e+00, 1.0622772013767669e+01, 1.2206883329864255e+01, 1.4027223820279268e+01,
    1.6119020948027551e+01, 1.8522755439841418e+01, 2.1284944674394648e+01, 2.4459042892590269e+01, 2.8106475650897377e+01, 3.2297828536610695e+01,
    3.7114213149203508e+01, 4.2648836782420439e+01, 4.9008806183799550e+01, 5.6317200298209784e+01, 6.4715452107403038e+01, 7.4366085659245940e+01,
    8.5455861253972017e+01, 9.8199389653503559e+01, 1.1284328525648564e+02, 1.2967093861180905e+02, 1.4900800062891784e+02, 1.7122868461604187e+02,
    1.9676300810421012e+02, 2.2610511460175258e+02, 2.5982283632295099e+02, 2.9856868294999509e+02, 3.4309246908417492e+02, 3.9425582475436693e+02,
    4.5304886979204105e+02, 5.2060937475742980e+02, 5.9824477922157064e+02, 6.8745749350484289e+02, 7.8997397351448706e+02, 9.0777813134110693e+02,
    1.0431497281803040e+03, 1.1987084925641964e+03, 1.3774648176845108e+03, 1.5828780189083436e+03, 1.8189232788935387e+03, 2.0901685758341446e+03,
    2.4018630835612289e+03, 2.7600387542291814e+03, 3.1716270494286800e+03, 3.6445930787218194e+03, 4.1880897414655765e+03, 4.8126348548959668e+03,
    5.5303146962783048e+03, 6.3550178980975252e+03, 7.3027042226581152e+03, 8.3917134174545281e+03, 9.6431201283206228e+03, 1.1081141738683076e+04,
    1.2733607027476588e+04, 1.4632494715248606e+04, 1.6814552320467552e+04, 1.9322007302220827e+04, 2.2203384251427578e+04, 2.5514443944955918e+04,
    2.9319262435339933e+04, 3.3691471058779483e+04, 3.8715681358217458e+04, 4.4489122496788041e+04, 5.1123522849071662e+04, 5.8747272182953282e+04,
    6.7507906274902969e+04, 7.7574962041275547e+04, 8.9143258438494697e+04, 1.0243666662452392e+05, 1.1771244234227551e+05, 1.3526620436579350e+05,
    1.5543765535274608e+05, 1.7861715581389511e+05, 2.0525327841984577e+05, 2.3586148883699448e+05, 2.7103412108532194e+05, 3.1145184046243853e+05,
    3.5789681586586579e+05, 4.1126785642601945e+05, 4.7259780537596118e+05, 5.4307352776633098e+05, 6.2405887883026747e+05, 7.1712109748508944e+05,
    8.2406113574115187e+05, 9.4694851095649926e+05, 1.0881613554026424e+06
    ])
_KEY201_SIN = np.array([
    -5.8602704468975832e-10, 4.8048608865691248e-09, -1.9771446411637653e-08, 5.5269671219669613e-08, -1.1943308954933238e-07, 2.1463221286467290e-07,
    -3.3618689168303137e-07, 4.7460375561491547e-07, -6.2010393774690030e-07, 7.6676687291580134e-07, -9.1382097135454670e-07, 1.0643058622431453e-06,
    -1.2227765259769829e-06, 1.3937672563527502e-06, -1.5810499935024131e-06, 1.7877429029287643e-06, -2.0163554732931835e-06, 2.2692361668040396e-06,
    -2.5484456753753459e-06, 2.8562454180065290e-06, -3.1946379472885138e-06, 3.5660627245815181e-06, -3.9725086879521357e-06, 4.4166880900049553e-06,
    -4.9004832484318760e-06, 5.4270061911978838e-06, -5.9978954164211254e-06, 6.6168927621162747e-06, -7.2851116079354270e-06, 8.0072717151050409e-06,
    -8.7834623074537534e-06, 9.6200272666869681e-06, -1.0515203893806995e-05, 1.1478097778214146e-05, -1.2503644637963446e-05, 1.3605711023452051e-05,
    -1.4773445946482213e-05, 1.6028992588028293e-05, -1.7351362255724533e-05, 1.8777046268328975e-05, -2.0267260460262344e-05, 2.1883367382220654e-05,
    -2.3555377916627267e-05, 2.5387811572432832e-05, -2.7256008695400006e-05, 2.9339419633021675e-05, -3.1417660496931868e-05, 3.3800354252141579e-05,
    -3.6099620853251731e-05, 3.8851364451479363e-05, -4.1374814496544362e-05, 4.4599458677506889e-05, -4.7332670414243943e-05, 5.1188885345556143e-05,
    -5.4081411276216648e-05, 5.8817223325521029e-05, -6.1748595688031771e-05, 6.7759540724914998e-05, -7.0477668174613142e-05, 7.8405578637928987e-05,
    -8.0416390943704734e-05, 9.1318327863896956e-05, -9.1689586953474345e-05, 1.0732811343723065e-04, -1.0434243728703050e-04, 1.2768616500544454e-04,
    -1.1822963938719640e-04, 1.5431866320930474e-04, -1.3280640984775213e-04, 1.9025153589598097e-04, -1.4674319163166278e-04, 2.4032678429752795e-04,
    -1.5722591368834836e-04, 3.1241839074880459e-04, -1.5869843609971545e-04, 4.1950672592888611e-04, -1.4061987157747521e-04, 5.8323092045218519e-04,
    -8.3489051752646859e-05, 8.3998754956665807e-04, 4.8166642275560894e-05, 1.2514134018618409e-03, 3.1809381315641577e-04, 1.9223958368384385e-03,
    8.3977376473052937e-04, 3.0319286374562394e-03, 1.8135564849093052e-03, 4.8856220017935168e-03, 3.5901028334771145e-03, 8.0038676866840078e-03,
    6.7764383041831900e-03, 1.3266048500372144e-02, 1.2406252926137538e-02, 2.2134422870406390e-02, 2.2191911862939541e-02, 3.6963935749189471e-02,
    3.8829525301567067e-02, 6.1308824849773652e-02, 6.6140947825550744e-02, 9.9797901699075597e-02, 1.0824048046010233e-01, 1.5616018092254505e-01,
    1.6533327222825650e-01, 2.2564430322919368e-01, 2.2062815143611000e-01, 2.7477185256445674e-01, 2.1146243753121693e-01, 2.0743923168031805e-01,
    2.9273136481935752e-03, -1.1875521395824359e-01, -4.8782254186329055e-01, -5.6383218637505039e-01, -7.0493717238569553e-01, -4.5783079000773212e-02,
    4.9572887642641605e-01, 1.2793243232592335e+00, 7.6984445533323119e-04, -1.0809022247360809e+00, -9.7826142685914264e-01, 2.4412168561816161e+00,
    -1.5742192933752286e+00, 1.0474779812704657e-01, 6.8117592802985916e-01, -8.1735467756558944e-01, 6.8894183180797697e-01, -5.2159909808793825e-01,
    3.8460793473043292e-01, -2.8421923628404017e-01, 2.1211949229810861e-01, -1.5993330303166020e-01, 1.2163508069999608e-01, -9.3164674694277416e-02,
    7.1777818840364910e-02, -5.5578938642792224e-02, 4.3227895023256115e-02, -3.3758552896064629e-02, 2.6463811011765734e-02, -2.0820205680643536e-02,
    1.6436812641415149e-02, -1.3019659797407497e-02, 1.0346416487448091e-02, -8.2481070073694342e-03, 6.5957434607522691e-03, -5.2904671538314026e-03,
    4.2562264095930237e-03, -3.4343067314751316e-03, 2.7792243881857633e-03, -2.2556298932160214e-03, 1.8359634980842480e-03, -1.4986732799921144e-03,
    1.2268558973329326e-03, -1.0072161412856525e-03, 8.2926783836384434e-04, -6.8471812797056032e-04, 5.6699154764304634e-04, -4.7086106657177131e-04,
    3.9216119658083232e-04, -3.2756429146830335e-04, 2.7440564055551230e-04, -2.3054635199842104e-04, 1.9426558644629048e-04, -1.6417564894421281e-04,
    1.3915493002405042e-04, -1.1829482010512426e-04, 1.0085758994667589e-04, -8.6242897629566575e-05, 7.3961097310559017e-05, -6.3611923024107272e-05,
    5.4867429384991716e-05, -4.7458310810428852e-05, 4.1162907677974120e-05, -3.5798353789658158e-05, 3.1213433790926005e-05, -2.7282808824825689e-05,
    2.3902339183058118e-05, -2.0985288286859604e-05, 1.8459236223596745e-05, -1.6263565819420051e-05, 1.4347411857118876e-05, -1.2667986114133353e-05,
    1.1189208624850740e-05, -9.8805899457079486e-06, 8.7163210286869178e-06, -7.6745372470670224e-06, 6.7367316776180222e-06, -5.8873002689814021e-06,
    5.1132081051238435e-06, -4.4037712527232031e-06, 3.7505516230790782e-06, -3.1473609977624753e-06, 2.5903624227147574e-06, -2.0782393943789611e-06,
    1.6123579385552092e-06, -1.1967503151651560e-06, 8.3762259929749212e-07, -5.4201864582008082e-07, 3.1540800169690699e-07, -1.5849870094735947e-07,
    6.4517602920670347e-08, -1.8937067758737377e-08, 3.0164202265996216e-09
    ])

# ============================================================
# 层状模型
# ============================================================
@dataclass
class LayeredModel:
    """1D TEM 层状模型（WLF carmod 系列接口兼容）。
    thickness_m: 长度 N，thickness_m[N-1] = np.inf 或 >=9999 代表半空间
    rho_ohm_m:  长度 N，各层电阻率 (Ω·m)
    IP 参数（可选，None=无极化）:
      chargeability: 长度 N，Pelton 极化率 m (0~1，通常 0.01~0.5)
      tau_s:         长度 N，时间常数 τ (s，通常 1e-4~10)
      c_exp:         长度 N，频率指数 c (0~1，通常 0.25~0.8，经典值0.5)
    """
    thickness_m: np.ndarray
    rho_ohm_m:   np.ndarray
    chargeability: Optional[np.ndarray] = None
    tau_s:         Optional[np.ndarray] = None
    c_exp:         Optional[np.ndarray] = None

    def __post_init__(self):
        self.thickness_m = np.atleast_1d(self.thickness_m).astype(np.float64)
        self.rho_ohm_m   = np.atleast_1d(self.rho_ohm_m).astype(np.float64)
        n = len(self.thickness_m)
        assert len(self.rho_ohm_m) == n, "thickness 与 rho 长度不一致"
        # 末层半空间：thickness→inf
        if self.thickness_m[-1] >= 9999:
            self.thickness_m = self.thickness_m.copy()
            self.thickness_m[-1] = np.inf
        # IP 参数校验：要么全None要么全有
        ip_attrs = (self.chargeability, self.tau_s, self.c_exp)
        n_none = sum(1 for x in ip_attrs if x is None)
        if n_none == 0:
            self.chargeability = np.atleast_1d(self.chargeability).astype(np.float64)
            self.tau_s         = np.atleast_1d(self.tau_s).astype(np.float64)
            self.c_exp         = np.atleast_1d(self.c_exp).astype(np.float64)
            for arr in (self.chargeability, self.tau_s, self.c_exp):
                assert len(arr) == n, "IP 参数长度与层数不一致"
        elif n_none == 3:
            pass
        else:
            raise ValueError("IP 三参数 chargeability/tau_s/c_exp 必须同时提供")

    @property
    def n_layers(self) -> int:
        return len(self.thickness_m)

    @property
    def has_ip(self) -> bool:
        return self.chargeability is not None

    # ------------------------------------------------------------------
    # 频率域：复电阻率（Cole-Cole Pelton 模型）
    # rho(w) = rho0 * (1 - m * (1 - 1/(1+(j w tau)^c)))
    # ------------------------------------------------------------------
    def complex_rho(self, omega: np.ndarray) -> np.ndarray:
        """omega: shape=(F,)  →  out: shape=(N, F) 复电阻率 (Ω·m)。"""
        omega = np.atleast_1d(omega).astype(np.complex128)
        rho0  = self.rho_ohm_m[:, None].astype(np.complex128)   # (N, 1)
        if not self.has_ip:
            return np.broadcast_to(rho0, (self.n_layers, omega.size)).copy()
        m  = self.chargeability[:, None]   # (N,1)
        tau = self.tau_s[:, None]          # (N,1)
        c  = self.c_exp[:, None]           # (N,1)
        jwt = (1j * omega[None, :]) * tau  # (N, F)
        # 数值稳定：jwt 太小→1/(1+x)≈1-x；jwt 太大→直接用幂
        with np.errstate(over='ignore', under='ignore'):
            zt = np.where(np.abs(jwt) < 1e-30, 1j, jwt) ** c
            term = 1.0 / (1.0 + zt)
        # 复电阻率
        return rho0 * (1.0 - m * (1.0 - term))


# ============================================================
# 1D TEM 正演核心: WLF Mod1 Boerner 扩散深度等效加权 (永远稳定, 三向自洽)
# ============================================================

def _layered_profile_on_zgrid(model: LayeredModel, z_fine: np.ndarray) -> tuple:
    """把 LayeredModel(层厚,ρ,IP三参数) 映射到深度细网格 z_fine 上。
    返回 (rho_z, m_z, tau_z, c_z)  每个 shape=(Nz,)."""
    thick = model.thickness_m.copy()
    if np.isinf(thick[-1]):
        # 半空间: 延伸至 2×网格底 (保证最深的细网格点仍落入半空间层内)
        thick[-1] = 2.0 * float(z_fine.max())
    edges = np.concatenate([[0.0], np.cumsum(thick)])
    idx = np.searchsorted(edges, z_fine, side='right') - 1
    idx = np.clip(idx, 0, model.n_layers - 1)
    rho_z = model.rho_ohm_m[idx]
    if model.has_ip:
        m_z   = model.chargeability[idx]
        tau_z = model.tau_s[idx]
        c_z   = model.c_exp[idx]
    else:
        m_z   = np.zeros_like(z_fine)
        tau_z = np.full_like(z_fine, 1e-3, dtype=np.float64)
        c_z   = np.full_like(z_fine, 0.5, dtype=np.float64)
    return rho_z, m_z, tau_z, c_z


def _diffusion_sensitivity_kernel(u: np.ndarray) -> np.ndarray:
    """WLF TEM 中心回线感应深度加权核 (VMD 归一化协方差)
       K(u) = 15/8 * u^2 * exp(-5u^2/2),   u = z / delta(t)
       峰值在 u ≈ 0.63; u > 4 → K ≈ 0.  (与 Raiche 1987 / WLF carmod1_kru 一致)"""
    u2 = np.clip(u * u, 0.0, 25.0)   # 截断: |u|>5 强制 0, 防 exp 下溢
    return 1.875 * u2 * np.exp(-2.5 * u2)


def _tau_eff_weighted(model: LayeredModel, t_arr: np.ndarray) -> np.ndarray:
    """扩散深度 K 核加权等效 Cole-Cole 时间常数 τ_eff(t) (与 tem_forward_1d 加权一致)。
    返回与 t_arr 等长的 τ_eff (s)。"""
    z_fine = np.geomspace(0.5, _Z_FINE_MAX_M, _Z_N_FINE)
    dz = np.gradient(z_fine)
    rho_z, _m_z, tau_z, _c_z = _layered_profile_on_zgrid(model, z_fine)
    out = np.zeros(len(t_arr), dtype=np.float64)
    for j, tj in enumerate(t_arr):
        rho_ref = float(np.exp(np.mean(np.log(np.clip(rho_z, 1e-2, 1e12)))))
        K = None
        for _ in range(3):
            delta = np.sqrt(2.0 * rho_ref * tj / _MU0)
            u = z_fine / max(delta, 1e-3)
            K = _diffusion_sensitivity_kernel(u) * dz
            K_sum = K.sum()
            if K_sum <= 1e-30:
                break
            rho_ref = K_sum / (K / np.clip(rho_z, 1e-3, None)).sum()
        if K is None or K.sum() <= 1e-30:
            out[j] = float(np.exp(np.mean(np.log(np.clip(tau_z, 1e-12, None)))))
        else:
            W = K / K.sum()
            out[j] = float(np.exp(np.dot(W, np.log(np.clip(tau_z, 1e-12, None)))))
    return out


def _pelton_rho_reducer(t_arr: np.ndarray,
                        m_eff: float, tau_eff: float, c_eff: float) -> np.ndarray:
    """Pelton Cole-Cole 电阻率乘子 (WLF pcvogoss_pc 正确方向: 在电阻率域应用, 不在 Vt 域)
       ρ_IP(t) = ρ_noIP(t) * R_IP(t), 其中
         R_IP(t) = 1 - m * (1 - 1/(1 + (t/τ)^c))   ∈ [1-m, 1]
       物理效果: 晚期 (t>>τ)  ρ_IP  = ρ*(1-m) → 更低阻 → 扩散更慢 → Vt 晚期更大。
       这与 TEM 野外实测的经典负激电视应 (ρa 晚期下降, Vt 尾部上翘) 一致。
    """
    t_safe = np.asarray(t_arr, dtype=np.float64)
    x = np.clip(t_safe / max(float(tau_eff), 1e-15), 1e-4, 1e5)
    term = 1.0 - 1.0 / (1.0 + np.power(x, float(c_eff)))
    m_clip = float(np.clip(m_eff, 0.0, 0.95))
    return 1.0 - m_clip * term


# ============================================================
# 对外统一入口: 时间门 TEM 正演 (冻结签名)
# ============================================================
def tem_forward_1d(model: LayeredModel,
                   tx_area_m2: float,
                   tx_current_a: float,
                   gates_sec: np.ndarray,
                   include_ip: bool = True,
                   n_freq: int = 140,   # 保留形参 (兼容旧签名), 本实现不用
                   log_fn: Optional[Callable[[str], None]] = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    WLF 标准 1D TEM 正演 (中心回线 VMD, dBz/dt 绝对值, 与预处理 S3-S7 量纲完全对齐).

    算法 (WLF Mod1 扩散深度等效):
      ① 深度细网格 z~geomspace(0.5m, 10km, 600), 映射 LayeredModel 至 (rho, m, tau, c)_z
      ② 每个 t_j:
         - 迭代 3 次自洽: rho_ref -> delta(t)=sqrt(2 rho_ref t/mu0)
         - K(z/delta) 加权 调和平均 → ρ_equiv(t) (无 IP 视电阻率真值)
         - K(z/delta) 加权 线性平均 → m_eff / exp(ΣW log τ) / c_eff
         - Nabighian 反向公式 (与 preprocess L278 严格代数互逆):
             Vt_noIP(t) = [ C * t^(-1.5) / ρ_equiv(t) ] ^ (3/2)
             C = (mu0/4π)^1.5 * (2*M)^(2/3) * RHOA_SCALE_CAL_DEFAULT
         - IP 校正: Vt_IP(t) = Vt_noIP(t) * k_Pelton(t, m_eff, tau_eff, c_eff)

    返回 (gates_sec, Vt), Vt ∈ (0, +∞).
    """
    gates_sec = np.atleast_1d(gates_sec).astype(np.float64)
    if not include_ip:
        # 临时禁用 IP (保持函数签名兼容)
        model = LayeredModel(model.thickness_m, model.rho_ohm_m, None, None, None)

    # ---- ① 深度细网格 ----
    z_fine = np.geomspace(0.5, _Z_FINE_MAX_M, _Z_N_FINE)
    dz = np.gradient(z_fine)
    rho_z, m_z, tau_z, c_z = _layered_profile_on_zgrid(model, z_fine)

    # ---- ② preprocess 对齐的 Nabighian 常数 ----
    mu = _MU0
    M = tx_current_a * tx_area_m2
    CAL = _RHOA_SCALE_CAL_DEFAULT
    COEF = (mu / (4.0 * np.pi)) ** 1.5 * (2.0 * M) ** (2.0 / 3.0) * CAL

    Vt_out = np.zeros_like(gates_sec, dtype=np.float64)
    for j, t in enumerate(gates_sec):
        t_safe = max(float(t), 1e-12)
        # 3 次自洽迭代: 初值 = 电阻率几何平均 → 稳定收敛
        rho_ref = float(np.exp(np.mean(np.log(np.clip(rho_z, 1e-2, 1e12)))))
        for _ in range(3):
            delta = np.sqrt(2.0 * rho_ref * t_safe / mu)
            u = z_fine / max(delta, 1e-3)
            K = _diffusion_sensitivity_kernel(u) * dz
            K_sum = K.sum()
            if K_sum <= 1e-30:
                break
            rho_over_unity = K / np.clip(rho_z, 1e-3, None)
            rho_ref = K_sum / rho_over_unity.sum()
        rho_equiv = max(rho_ref, 1e-4)

        # IP 加权平均: m 线性, τ→exp(加权 logτ), c 线性
        m_eff = 0.0; tau_eff = 1e-3; c_eff = 0.5
        if model.has_ip and (K_sum > 1e-30):
            W = K / K_sum
            m_eff = float(np.dot(W, m_z))
            log_tau_avg = float(np.dot(W, np.log(np.clip(tau_z, 1e-12, None))))
            tau_eff = float(np.exp(log_tau_avg))
            c_eff = float(np.dot(W, c_z))

        # ---- (关键点) IP 在电阻率域应用 Pelton, 再求 Vt (正确物理方向) ----
        rho_for_vt = rho_equiv
        if include_ip and model.has_ip:
            R_ip = _pelton_rho_reducer(np.array([t_safe]), m_eff, tau_eff, c_eff)[0]
            rho_for_vt = rho_equiv * max(float(R_ip), 1e-4)

        # ---- Vt (与 preprocess S3 S7 严格代数互逆) ----
        vt_base = (COEF * (t_safe ** (-1.5)) / rho_for_vt) ** 1.5

        Vt_out[j] = max(float(vt_base), 1e-35)

    if log_fn:
        med_rho_equiv = (COEF * (np.median(gates_sec) ** (-1.5)) / np.median(Vt_out) ** (2.0/3.0))
        log_fn(f"  [FW-1D] N_l={model.n_layers} IP={model.has_ip} Ng={len(gates_sec)} "
               f"Vt_median={np.median(Vt_out):.2e} rhoa_equiv(med_t)={med_rho_equiv:.1f} Ohm*m")
    return gates_sec, Vt_out


# ============================================================
# 视电阻率 (与 preprocess S7 Nabighian 公式 100% 对齐, 禁止擅自修改)
# ============================================================
def apparent_resistivity_dbzdt(dbzdt: np.ndarray, t_sec: np.ndarray,
                               tx_area_m2: float, tx_current_a: float,
                               RHOA_SCALE_CAL: float = 1_000_000) -> np.ndarray:
    """与 WLF RhoCalc / preprocess.py S7 完全一致的 Nabighian 中心回线 ρa 公式。
       rhoa = (mu0/4π)^1.5 * (2*M)^(2/3) * RHOA_SCALE_CAL * t^(-1.5) / |Vt|^(2/3)
       — 与 tem_forward_1d 为严格代数互逆, 因此正演→视电阻率在均匀半空上恒等。"""
    t_safe = np.maximum(np.asarray(t_sec, dtype=np.float64), 1e-12)
    vt = np.maximum(np.abs(np.asarray(dbzdt, dtype=np.float64)), 1e-35)
    mu = _MU0
    M = tx_current_a * tx_area_m2
    COEF = (mu / (4.0 * np.pi)) ** 1.5 * (2.0 * M) ** (2.0 / 3.0) * RHOA_SCALE_CAL
    rhoa = COEF * (t_safe ** (-1.5)) / (vt ** (2.0 / 3.0))
    return np.asarray(rhoa, dtype=np.float64)


# ============================================================
# 2D/3D 占位接口（返回结构化错误信息；真正调用走 M7_WLF_Bridge）
# ============================================================
def tem_forward_2d_section(stations_x_m: np.ndarray,
                           model_2d: dict,
                           gates_sec: np.ndarray,
                           backend: str = 'wlf_swp24',
                           log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """2D 剖面正演占位（按 WLF 设计原理：carmod2_kru → 实际执行 SWP24.EXE 桥接）。
    backend='python' 将抛出 NotImplementedError(纯Python 2D FDTD 计算量过大，建议 WLF EXE)。
    """
    if backend == 'python':
        raise NotImplementedError("纯Python 2D TEM FDTD 计算量极大(≥10⁷网格)，"
                                  "请使用 backend='wlf_swp24' 桥接 WLF SWP24D.EXE")
    return {'status': 'bridge_ready',
            'exe_hint': 'SWP24D.EXE',
            'stations_x_m': stations_x_m,
            'n_stations': len(stations_x_m),
            'n_gates': len(gates_sec),
            'message': f'请通过 M7_WLF_Bridge.run_2d_inv_wlf(...) 调用 WLF 2D 反演'}


# ============================================================
# 真频域正演: 复频率域层状递推 + Guptasarma J1 + Key(2012) 正弦DLF 频→时转换
#   ① 复频率 ω 网格上, 含空气界面的层状反射系数递推 → Hz_sec(iω) (中心回线 J1 核)
#   ② 阶跃关断: dBz/dt(t) = -(2μ0/π) ∫₀^∞ Im[Hz_sec(ω)] sin(ωt) dω
#      Key(2012) 201点正弦DLF: ∫ f(ω)sin(ωt)dω ≈ (1/t) Σ wsin_i f(base_i/t)
#   注: Gaver-Stehfest 实数s域逆变换对二次场失效 (O(s) 解析项淹没 s^{3/2} 物理项,
#       数值上无法分离, 已弃用), 复频率ω域 + 正弦DLF 为标准可靠路径。
# ============================================================

def _refl_coef_freq(lam: np.ndarray, omega: np.ndarray,
                    rho: np.ndarray, thick: np.ndarray,
                    mu0: float) -> np.ndarray:
    """复频率域层状介质反射系数递推 R*(λ, iω)（自底层半空间向上, 向量化）。
    lam: (Nλ,), omega: (Nω,)
    rho: (Nlayer,) float 或 (Nlayer, Nω) complex（频率域 Cole-Cole 复电阻率, IP）
    thick: (Nlayer,)（rho=inf 表示空气层）。
    返回 R*: shape (Nω, Nλ)（顶层界面向下看的总反射系数）。"""
    lam_sq = (lam ** 2)[None, None, :]                         # (1, 1, Nλ): λ²
    if np.iscomplexobj(rho) and np.asarray(rho).ndim == 2:
        # 频率域复电阻率: ρ_i(ω) → u_i(ω,λ), 广播 (Nω, Nlayer, 1)
        rho_t = np.asarray(rho).T[:, :, None]                  # (Nω, Nlayer, 1)
        u_all = np.sqrt(lam_sq + 1j * omega[:, None, None] * mu0 / rho_t)
    else:
        u_all = np.sqrt(lam_sq + 1j * omega[:, None, None] * mu0
                        / np.asarray(rho)[None, :, None])   # (Nω, Nlayer, Nλ)
    R = np.zeros((omega.size, lam.size), dtype=np.complex128)
    # 从底层半空间向上递推 (层索引: 0=空气层置顶)
    for i in range(len(thick) - 2, -1, -1):
        u_i  = u_all[:, i, :]
        u_i1 = u_all[:, i + 1, :]
        p = (u_i - u_i1) / (u_i + u_i1)
        h = float(thick[i])
        e = 0.0 if np.isinf(h) else np.exp(-2.0 * u_i * h)     # h=0 (空气层) → e=1
        R = (p + R * e) / (1.0 + p * R * e)
    return R


def _hz_freq(omega: np.ndarray, model: LayeredModel,
             tx_area: float, tx_current: float) -> np.ndarray:
    """复频率域二次磁场 Hz_sec(iω) — 中心回线装置（等效圆回线, 向量化）。

    核函数 (Ward & Hohmann 1988；AarhusInv/LEROI 同式):
        Hz_sec(ω) = (I·a/2) ∫₀^∞ λ·R*(λ,iω)·J1(λa) dλ,   a = sqrt(tx_area/π)
      R*: 含空气/大地界面的总反射系数（空气层 ρ=∞, h=0 置顶递推）
    一次场与 ω 无关, 对阶跃关断的 dBz/dt 无贡献, 仅保留二次场。
    IP: 若 model 含 Pelton 三参数, 自动走频率域复电阻率 ρ(ω) 路径 (严格物理)。

    omega: (Nω,) → 返回 (Nω,) 复数 Hz_sec [A/m]。"""
    a = np.sqrt(tx_area / np.pi)
    if a <= 0:
        raise ValueError('tx_area 必须为正')
    lam = _GUPT47_BASE / a                       # λ_i = base_i / a
    # 空气层置顶: ρ_air=inf → u_air=λ; h_air=0 → e=1, 递推直接给出地表 R*
    if model.has_ip:
        rho_eff = model.complex_rho(omega)       # (Nlayer, Nω) 复电阻率
        rho_air = np.full((1, omega.size), np.inf)
        rho_full = np.concatenate([rho_air, rho_eff], axis=0)   # (Nlayer+1, Nω)
    else:
        rho_full = np.concatenate([[np.inf], model.rho_ohm_m])
    thick_full = np.concatenate([[0.0], model.thickness_m])
    R_star = _refl_coef_freq(lam, omega, rho_full, thick_full, _MU0)
    f_lam = R_star * lam[None, :]                # (Nω, Nλ): λ·R*
    # ∫ f J1(λa) dλ ≈ (1/a) Σ w_i f(base_i/a)
    integral = f_lam @ _GUPT47_J1 / a            # (Nω,)
    return (tx_current * a / 2.0) * integral     # = (I/2)·Σ w_i λ_i R*_i


def tem_forward_1d_freq(model: LayeredModel,
                        tx_area_m2: float,
                        tx_current_a: float,
                        gates_sec: np.ndarray,
                        r_off_m: Optional[float] = None,
                        log_fn: Optional[Callable[[str], None]] = None,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """真频域 1D TEM 正演 (复频率域层状递推 + Guptasarma J1 + Key2012 正弦DLF)。

    与现有 tem_forward_1d (扩散深度近似) 接口完全对齐:
      input:  model, tx_area, tx_current, gates_sec
      output: (t_sec, dbzdt)   [T/s]
      r_off_m: 已废弃（中心回线用等效回线半径 a=√(A/π)，无奇异），保留仅为签名兼容。

    原理:
      ① 复频率域: 含空气界面的层状反射系数递推 → Hz_sec(iω)
         (中心回线核, Guptasarma & Singh 47 点 J1 数字滤波, λ 网格)
      ② 频→时转换: Key (2012) 201 点正弦 DLF
         dBz/dt(t) = -(2μ0/π)·(1/t)·Σ wsin_k·Im[Hz_sec(base_k/t)]
      ③ IP (激电效应): 频率域复电阻率路径 — model.has_ip 时以 Pelton Cole-Cole
         ρ(ω)=ρ₀(1-m(1-1/(1+(jωτ)^c))) 逐层代入反射系数递推 (_refl_coef_freq /
         _hz_freq 支持 (Nlayer, Nω) 复电阻率广播), 频→时转换后 IP 效应自然进入时域。
         晚期 blend: t≤0.5τ_eff 保留 DLF IP 全响应, t≥2τ_eff 回归无 IP 基线
         (DLF 宽频相消在色散扰动下产生晚期伪影, 详见函数内注释与 _diag_ip_late.py;
         τ_eff 为扩散深度 K 核加权等效时间常数)。
      ④ 输出真 SI dBz/dt [T/s]: 晚期与 Ward & Hohmann t^(-5/2) 渐近精确一致
         (已验证: 100Ω·m 半空间 a/δ<0.5 时晚期 ρa 恢复 100±1%)。
         视电阻率必须用 apparent_resistivity_dbzdt_freq (真晚期公式);
         勿复用 apparent_resistivity_dbzdt (WLF 体系 t^(-2.25) 等效量纲, 不兼容)。

    冻结函数签名 (GUI / 反演模块依赖):
      (t, dbz) = tem_forward_1d_freq(model, tx_area, tx_current, gates_sec)
    """
    t = np.asarray(gates_sec, dtype=np.float64)
    n = len(t)
    dbzdt = np.zeros(n, dtype=np.float64)

    for i, ti in enumerate(t):
        omega = _KEY201_BASE / ti
        # model.has_ip 时 _hz_freq 内部走 complex_rho(ω) 复电阻率路径
        Hz = _hz_freq(omega, model, tx_area_m2, tx_current_a)
        # dBz/dt(t) = (2μ0/π)·(1/t)·Σ wsin_k·Im[Hz_sec(base_k/t)]  (整体符号无关, 末取 abs)
        dbzdt[i] = (2.0 * _MU0 / np.pi) / ti * float(np.sum(_KEY201_SIN * Hz.imag))

    if model.has_ip:
        # IP blend — 依据 ( Watson 渐近分析 + 数值实验, 详见 _diag_ip_late.py ):
        #   ① 频域复电阻率路径严格正确 (低频一阶展开 m·√(ωτ/2) 逐频点验证吻合);
        #   ② 早期 DLF (t ≲ 0.5τ) 可靠: IP 谷 ratio@0.2τ 与 ρ_eff(ω~1/t) 理论吻合 (0.815 vs 0.81);
        #   ③ 晚期 dBz/dt 是 ~11 个量级的宽频相消残差, Cole-Cole 色散扰动破坏
        #      DLF 相消平衡 → 晚期伪影 (t=2.6τ 时已 +39%, t=100τ 时 |dBz/dt| 虚低 7.7 倍);
        #   ④ DLF 为线性算子, 差分/分响应方案与全响应恒等, 无法绕过伪影。
        # 故: t≤0.5τ_eff 保留 DLF IP 全响应; t≥2τ_eff 回归无 IP 基线 (系统偏差 ~IP 物理效应
        # 量级, 远小于伪影); 中间 smoothstep 过渡。适用范围: IP 反演灵敏区 t~τ 附近。
        noip = LayeredModel(model.thickness_m, model.rho_ohm_m)
        dbz_noip = np.empty(n, dtype=np.float64)
        for i, ti in enumerate(t):
            omega = _KEY201_BASE / ti
            Hz0 = _hz_freq(omega, noip, tx_area_m2, tx_current_a)
            dbz_noip[i] = (2.0 * _MU0 / np.pi) / ti * float(np.sum(_KEY201_SIN * Hz0.imag))
        tau_eff = _tau_eff_weighted(model, t)
        x = np.log(np.maximum(t, 1e-12) / np.maximum(tau_eff, 1e-12))
        w = np.clip((x - np.log(0.5)) / (np.log(2.0) - np.log(0.5)), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)                      # smoothstep 0→1
        ratio = np.abs(dbzdt) / np.maximum(np.abs(dbz_noip), 1e-300)
        dbzdt = dbz_noip * ((1.0 - w) * ratio + w)

    # 输出真 SI dBz/dt [T/s] (取绝对值; 符号仅取决于发射电流方向约定)
    # 量纲验证: 晚期与 W&H 渐近一致, ρa 用 apparent_resistivity_dbzdt_freq 反解
    return t, np.abs(dbzdt)


def apparent_resistivity_dbzdt_freq(dbzdt: np.ndarray, t_sec: np.ndarray,
                                    tx_area_m2: float, tx_current_a: float,
                                    RHOA_SCALE_CAL: float = 1_000_000) -> np.ndarray:
    """真频域正演配套视电阻率 — Ward & Hohmann (1988) 中心回线晚期渐近严格反解。

    晚期渐近 (t→∞, a/δ≪1):
        dBz/dt = 3·I·a²·μ0^2.5 / (20·π^1.5·ρ^1.5·t^2.5),   a = √(tx_area/π)
    反解:
        ρa = [3·I·a²·μ0^2.5 / (20·π^1.5·t^2.5·|dbzdt|)]^(2/3)

    均匀半空间晚期 (a/δ < 0.5) 精确恢复真电阻率 (验证 100Ω·m → 99.3~102);
    早期 ρa 上翘 (∝ t^(-5/3)) 为晚期视电阻率定义的固有物理行为, 非误差。
    RHOA_SCALE_CAL 形参仅为签名兼容保留 (真 SI 公式不使用校准因子)。"""
    t_safe = np.maximum(np.asarray(t_sec, dtype=np.float64), 1e-12)
    d = np.maximum(np.abs(np.asarray(dbzdt, dtype=np.float64)), 1e-35)
    a = np.sqrt(float(tx_area_m2) / np.pi)
    num = 3.0 * float(tx_current_a) * a * a * _MU0 ** 2.5
    den = 20.0 * np.pi ** 1.5 * t_safe ** 2.5 * d
    return (num / den) ** (2.0 / 3.0)


def tem_forward_1d_freq_vt(model: LayeredModel,
                           tx_area_m2: float,
                           tx_current_a: float,
                           gates_sec: np.ndarray,
                           include_ip: bool = True,
                           log_fn: Optional[Callable[[str], None]] = None,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """真频域正演 → WLF Vt 量纲封装 (反演/GUI 可直接替换 tem_forward_1d)。

    链路:
      ① tem_forward_1d_freq             → 真 SI dBz/dt [T/s] (IP 走频率域复电阻率)
      ② apparent_resistivity_dbzdt_freq → W&H 真晚期 ρa(t)
      ③ WLF Nabighian 正向公式代数反映射回 Vt:
            Vt(t) = [ COEF·t^(-1.5) / ρa(t) ]^1.5,
            COEF = (μ0/4π)^1.5 · (2M)^(2/3) · RHOA_SCALE_CAL_DEFAULT
        输出量纲与 tem_forward_1d 严格一致 (验证: 无 IP 模型晚期两引擎
        ρa 一致到 0.6~1.1% @ t>6ms, Vt 量纲映射无系统偏移)。

    用途: 调用侧 (tfm_gui / ip_inversion) 将 tem_forward_1d 替换为本函数
    即升级为真频域正演核 — 观测 Vt 量纲与 preprocess 提取对齐, 其余不变。
    include_ip=False 时临时禁用 IP (签名兼容 tem_forward_1d)。"""
    gates_sec = np.atleast_1d(gates_sec).astype(np.float64)
    if not include_ip and model.has_ip:
        model = LayeredModel(model.thickness_m, model.rho_ohm_m, None, None, None)

    t, dbzdt = tem_forward_1d_freq(model, tx_area_m2, tx_current_a, gates_sec)
    rhoa = apparent_resistivity_dbzdt_freq(dbzdt, t, tx_area_m2, tx_current_a)
    M = float(tx_current_a) * float(tx_area_m2)
    COEF = (_MU0 / (4.0 * np.pi)) ** 1.5 * (2.0 * M) ** (2.0 / 3.0) * _RHOA_SCALE_CAL_DEFAULT
    vt = (COEF * t ** (-1.5) / np.maximum(rhoa, 1e-4)) ** 1.5
    return t, np.maximum(vt, 1e-35)


def tem_forward_3d_volume(grid_xyz: tuple, model_3d: dict,
                          gates_sec: np.ndarray,
                          backend: str = 'wlf_carmod3',
                          log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """3D 体积正演占位（按 WLF carmod3_kru → 3D 层析切片）。"""
    if backend == 'python':
        raise NotImplementedError("纯Python 3D TEM Yee-lattice FDTD 计算量极大，"
                                  "请使用 WLF 商业 EXE 桥接")
    return {'status': 'bridge_ready',
            'exe_hint': 'TRANS24.exe + 2D结果插值→3D',
            'grid': grid_xyz, 'n_gates': len(gates_sec)}


__all__ = ['LayeredModel', 'tem_forward_1d', 'apparent_resistivity_dbzdt',
           'tem_forward_1d_freq', 'apparent_resistivity_dbzdt_freq',
           'tem_forward_1d_freq_vt',
           'tem_forward_2d_section', 'tem_forward_3d_volume',
           '_MU0', '_GUPT61_BASE', '_GUPT61_J0', '_GUPT47_BASE', '_GUPT47_J1',
           '_KEY201_BASE', '_KEY201_SIN']
