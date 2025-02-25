# CompAstro_2024
Computational Astrophysics course project modeling stellar energy transport.
\begin{equation}
    \frac{\partial r}{\partial m} = \frac{1}{4 \pi r^2 \rho}
\end{equation}
\begin{equation}
    \frac{\partial P}{\partial m} = -\frac{Gm}{4 \pi r^2}
\end{equation}
\begin{equation}
    \frac{\partial L}{\partial m} = \epsilon
\end{equation}
\begin{equation}
\frac{\partial T}{\partial m} =
\begin{cases}
    \nabla^* \frac{T}{P} \frac{\partial P}{\partial m} & \text{if } \nabla_\text{stable} > \nabla_\text{ad} \\
    \frac{3\kappa L}{256 \pi^2 \sigma r^4 T^3} & \text{otherwise}
\end{cases}
\end{equation}
\begin{equation}
    P = \frac{4 \sigma}{3c}T^4 + \frac{\rho k_{B}}{\mu m_{u}}T
\end{equation}
\begin{equation}
    \rho = (P - \frac{4 \sigma T^4}{3c}) \frac{\mu m_{u}}{k_{B}T}
\end{equation}
