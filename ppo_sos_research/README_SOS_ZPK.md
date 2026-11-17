# Continuous Controller to Discrete SOS–ZPK Realization

## 1. Continuous Controller

Consider the SISO controller

$$
K(s)=\frac{N(s)}{D(s)}.
$$

## 2. Discretization and SOS–ZPK Representation

Using `c2d(K, Ts, 'zoh')`, we obtain

$$
K_d(z)=
\frac{b_0+b_1z^{-1}+\cdots+b_nz^{-n}}
     {1+\alpha_1z^{-1}+\cdots+\alpha_nz^{-n}}.
$$

For relative degree one, $b_0=0$ and $b_1\neq0$.

The SOS decomposition groups the roots into sections with at most two poles, two zeros, and a gain. Local gains are collected into the global gain $K_g$, leaving normalized sections:

$$
H_i(z)=
\frac{\beta_0+\beta_1z^{-1}+\beta_2z^{-2}}
     {1+a_1z^{-1}+a_2z^{-2}}.
$$

For a biquad with poles $p_1,p_2$:

| Finite zeros | $\beta_1$ | $\beta_2$ | $a_1$ | $a_2$ |
|---|---|---|---|---|
| $\zeta_1,\zeta_2$ | $-(\zeta_1+\zeta_2)$ | $\zeta_1\zeta_2$ | $-(p_1+p_2)$ | $p_1p_2$ |
| $\zeta_1$ | $1$ | $-\zeta_1$ | $-(p_1+p_2)$ | $p_1p_2$ |

The normalized direct term is

$$
\beta_0=
\begin{cases}
1, & \text{equal-degree section},\\
0, & \text{strictly proper section}.
\end{cases}
$$

The second row already includes one factor $z^{-1}$.

## 3. Direct Form II: Local State-Space Realization

Direct Form II uses two delay elements per biquad, compared with four in a conventional Direct Form I implementation:

$$
w_i[k]
=
v_{i-1}[k]-a_1w_i[k-1]-a_2w_i[k-2],
$$

$$
v_i[k]
=
\beta_0w_i[k]+\beta_1w_i[k-1]+\beta_2w_i[k-2].
$$

Define the local state as

$$
x_i[k]=
\begin{bmatrix}
w_i[k-1]\\
w_i[k-2]
\end{bmatrix}.
$$

The controllable canonical realization is

$$
A_i=
\begin{bmatrix}
-a_1 & -a_2\\
1 & 0
\end{bmatrix},
\qquad
B_i=
\begin{bmatrix}
1\\
0
\end{bmatrix},
$$

$$
C_i=
\begin{bmatrix}
\beta_1-\beta_0a_1 &
\beta_2-\beta_0a_2
\end{bmatrix},
\qquad
D_i=\beta_0.
$$

Therefore,

$$
x_i[k+1]=A_ix_i[k]+B_iv_{i-1}[k],
$$

$$
v_i[k]=C_ix_i[k]+D_iv_{i-1}[k].
$$

A first-order section uses only one state.

## 4. Cascade State-Space Realization

For two sections, let

$$
v_0[k]=y[k],
\qquad
u[k]=K_gv_2[k],
\qquad
x_{\mathrm{cas}}[k]=
\begin{bmatrix}
x_1[k]\\
x_2[k]
\end{bmatrix}.
$$

The equivalent cascade realization is

$$
A_{\mathrm{cas}}=
\begin{bmatrix}
A_1 & 0\\
B_2C_1 & A_2
\end{bmatrix},
\qquad
B_{\mathrm{cas}}=
\begin{bmatrix}
B_1\\
B_2D_1
\end{bmatrix},
$$

$$
C_{\mathrm{cas}}=
K_g
\begin{bmatrix}
D_2C_1 & C_2
\end{bmatrix},
\qquad
D_{\mathrm{cas}}=K_gD_2D_1.
$$

Finally,

$$
x_{\mathrm{cas}}[k+1]
=
A_{\mathrm{cas}}x_{\mathrm{cas}}[k]
+B_{\mathrm{cas}}y[k],
$$

$$
u[k]
=
C_{\mathrm{cas}}x_{\mathrm{cas}}[k]
+D_{\mathrm{cas}}y[k].
$$

Each section passes its output to the next section within the same sample. The cascade preserves the delays already encoded in the sections.
