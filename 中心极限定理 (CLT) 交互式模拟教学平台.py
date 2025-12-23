import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom, geom, chi2, t, f, poisson, expon, uniform, skew
import platform

# --- 1. 环境配置与中文适配 ---
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="高级 CLT 仿真平台", layout="wide")

st.title("📊 中心极限定理 (CLT) 深度交互仿真平台")
st.markdown("本系统通过自定义母体分布参数，演示均值序列如何向正态分布收敛。")

# --- 2. 侧边栏：参数输入模块 ---
st.sidebar.header("🔧 配置模拟参数")

dist_list = [
    "均匀分布 (Uniform)", 
    "泊松分布 (Poisson)", 
    "指数分布 (Exponential)", 
    "正态分布 (Normal)", 
    "0-1 分布 (Bernoulli)", 
    "二项分布 (Binomial)", 
    "几何分布 (Geometric)", 
    "卡方分布 (Chi-Square)", 
    "t 分布", 
    "F 分布"
]

dist_type = st.sidebar.selectbox("选择母体分布类型", dist_list)

# --- 3. 动态分布参数设置 (核心修复：使用关键词匹配确保框体弹出) ---
st.sidebar.subheader("母体分布自身参数")

params = {} 

# 这里的逻辑通过 'in' 关键字匹配，防止因为空格或括号导致的匹配失败
if "均匀" in dist_type:
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        params['a'] = st.number_input("区间下限 a", value=0.0, key="uni_a")
    with col_b:
        params['b'] = st.number_input("区间上限 b", value=10.0, key="uni_b")
    if params['a'] >= params['b']:
        st.sidebar.error("错误：下限 a 必须小于上限 b")

elif "泊松" in dist_type:
    params['theta'] = st.sidebar.number_input("参数 θ (Lambda)", min_value=0.1, value=3.0, step=0.5, key="poi_l")

elif "指数" in dist_type:
    params['lambda'] = st.sidebar.number_input("参数 λ (Rate)", min_value=0.01, value=1.0, step=0.1, key="exp_r")

elif "正态" in dist_type:
    col_mu, col_std = st.sidebar.columns(2)
    with col_mu:
        params['mu'] = st.number_input("均值 μ", value=0.0, key="norm_u")
    with col_std:
        params['sigma'] = st.number_input("标准差 σ", min_value=0.01, value=1.0, key="norm_s")

elif "0-1" in dist_type:
    params['p'] = st.sidebar.slider("成功概率 p", 0.0, 1.0, 0.5, key="bern_p")

elif "二项" in dist_type:
    params['n_trial'] = st.sidebar.slider("试验次数 n", 1, 100, 10, key="bin_n")
    params['p'] = st.sidebar.slider("成功概率 p", 0.0, 1.0, 0.5, key="bin_p")

elif "卡方" in dist_type:
    params['df'] = st.sidebar.slider("自由度 df", 1, 50, 5, key="chi_df")

elif "t 分布" in dist_type:
    params['df'] = st.sidebar.slider("自由度 df", 1, 100, 10, key="t_df")

elif "F 分布" in dist_type:
    params['dfn'] = st.sidebar.slider("分子自由度 dfn", 1, 100, 10, key="f_dfn")
    params['dfd'] = st.sidebar.slider("分母自由度 dfd", 1, 100, 20, key="f_dfd")

# 核心抽样参数调节
st.sidebar.subheader("CLT 抽样规模控制")
n_sample = st.sidebar.slider("单次样本容量 (n)", 1, 5000, 30)
N_sim = st.sidebar.slider("总模拟抽样次数 (N)", 100, 10000, 2000)

# --- 4. 核心计算模块 ---
def get_sample_means(dist, p, n, N):
    # 使用 np.mean(..., axis=1) 实现矩阵化快速运算
    if "均匀" in dist:
        data = uniform.rvs(loc=p['a'], scale=p['b'] - p['a'], size=(N, n))
    elif "泊松" in dist:
        data = poisson.rvs(mu=p['theta'], size=(N, n))
    elif "指数" in dist:
        data = expon.rvs(scale=1/p['lambda'], size=(N, n))
    elif "正态" in dist:
        data = norm.rvs(loc=p['mu'], scale=p['sigma'], size=(N, n))
    elif "0-1" in dist:
        data = bernoulli.rvs(p['p'], size=(N, n))
    elif "二项" in dist:
        data = binom.rvs(p['n_trial'], p['p'], size=(N, n))
    elif "几何" in dist:
        data = geom.rvs(0.5, size=(N, n))
    elif "卡方" in dist:
        data = chi2.rvs(df=p['df'], size=(N, n))
    elif "t 分布" in dist:
        data = t.rvs(df=p['df'], size=(N, n))
    elif "F 分布" in dist:
        data = f.rvs(dfn=p['dfn'], dfd=p['dfd'], size=(N, n))
    else:
        return np.zeros(N)
    return np.mean(data, axis=1)

# 执行计算并渲染
try:
    means = get_sample_means(dist_type, params, n_sample, N_sim)

    # --- 5. 可视化渲染模块 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(means, bins=60, density=True, alpha=0.6, color='#1f77b4', label='样本均值观测分布')
    
    # 叠加正态拟合曲线（理论对比）
    mu_fit, std_fit = norm.fit(means)
    x_range = np.linspace(min(means), max(means), 200)
    y_pdf = norm.pdf(x_range, mu_fit, std_fit)
    ax.plot(x_range, y_pdf, 'r--', lw=2, label=f'理论正态拟合曲线\n($\mu={mu_fit:.2f}, \sigma={std_fit:.2f}$)')
    
    ax.set_title(f"{dist_type} 母体在 n={n_sample} 时的均值分布", fontsize=14)
    ax.set_xlabel("均值数值")
    ax.set_ylabel("概率密度")
    ax.legend()
    st.pyplot(fig)

    # --- 6. 统计面板 ---
    st.subheader("📋 统计指标分析")
    c1, c2, c3 = st.columns(3)
    c1.metric("观测均值 (Mean)", f"{mu_fit:.4f}")
    c2.metric("观测标准误 (Std Error)", f"{std_fit:.4f}")
    c3.metric("分布偏度 (Skewness)", f"{skew(means):.4f}")

except Exception as e:
    st.info("💡 请在左侧配置母体分布参数以开始模拟。")

st.info("💡 专利技术说明：本系统通过『关键词驱动的动态参数映射逻辑』，解决了多类型概率分布在 Web 交互环境下的参数配置冲突问题。")
