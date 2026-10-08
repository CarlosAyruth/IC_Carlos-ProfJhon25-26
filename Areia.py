# Streamlit para abrir um dashboard interativo: Usar comando no terminal de python -m streamlit run Areia.py

import plotly.express as px
import numpy as np
import math
from CoolProp.CoolProp import PropsSI
import pandas as pd
import streamlit as st
from scipy.optimize import newton


# Configuração do dashboard
st.set_page_config(page_title="Dashboard - Iniciação Científica 2025/2026", layout="wide")
st.title("Comparação de Modelos Semi-empíricos para Estimativa do Rendimento Operacional de um Recuperador de Calor de Leito Fluidizado")
st.markdown("Gráficos em função do diâmetro da partícula  \n Partícula: Areia")

# Cálculo do fator de atrito
def Colebrook(f, Re_i, Di):
    rugosidade_relativa = 0.000001/Di
    return (1.0 / np.sqrt(f)) + 2.0 * np.log10(rugosidade_relativa / 3.7 + 2.51 / (Re_i * np.sqrt(f)))

# Cálculo do número de Nusselt interno
def Nusselt(Pr_i, Re_i, Di):

    if Re_i < 2300:
        
        Regime = "Laminar"
        Nu_i = 3.66

    elif Re_i >= 2300 and Re_i <= 4500:

        Regime = "Transição"
        fa = 3.03*(10**-12)*(Re_i**3) - 3.67*(10**-8) * \
            (Re_i**2) + 0.000146*Re_i - 0.151
        Nu_i = (fa/8)*(Re_i-1000) * \
            (Pr_i/(1+(12.7*((fa/8)**0.5)*((Pr_i**(2/3))-1))))

    elif Re_i > 4500:

        Regime = "Turbulento"
        f = 0.01
        fa = newton(Colebrook, x0=f, args=(Re_i, Di))
        Nu_i = (fa/8)*(Re_i-1000) * \
            (Pr_i/(1+(12.7*((fa/8)**0.5)*((Pr_i**(2/3))-1))))

        if Pr_i > 2000 or Pr_i < 0.5 or Re_i > 5*(10**6):

            print("Inválido")

    return Nu_i, Regime

# Cálculo da condutividade térmica da parede dos tubos
def CondutividadeMaterial(T_filme):

    # Avaliado para o aço inox 304 https://steelprogroup.com/pt/stainless-steel/properties/thermal-conductivity/
    T = [293, 373, 573, 773]  # Temperatura [k]
    K = [16.2, 16.2, 18.4, 21.5]  # Condutividade térmica [W/m*K]

    Valor_K_t = np.interp(T_filme, T, K)

    return Valor_K_t
# Correlação de Molerus (1997)
def Molerus(e_mf, rho_g, U_mf, U_g, Ar, rho_p, Visc_g, k_p,  K_g, cp_p, g, Pr_g, T, De, D_p, cp_g, P1, A, zeta, **kwargs):

    Ll = (Visc_g / (math.sqrt(g) * (rho_p - rho_g)))**(2/3)

    PI_2 = K_g/(2*cp_p*Visc_g)
    PI_3 = cp_g*Visc_g/K_g
    PI_4 = rho_g/(rho_p-rho_g)
    PI_5 = (U_g-U_mf)*((rho_p*cp_p/(K_g*g))**(1/3))
    PI_6 = (U_mf)*((rho_p*cp_p/(K_g*g))**(1/3))
    PI_7 = 1-e_mf

    Nu_pc = (0.125*PI_7/(1+PI_2*(1+0.28*(PI_7**2)*(PI_4**0.5)*PI_5*PI_6)))*((1+33.3*((PI_6/PI_5)**(1/3))*(PI_5**-1))**-1)

    Nu_gc = 0.165*((PI_3*PI_4)**(1/3))*((1+0.05*PI_6/PI_5)**-1)

    Coef_Molerus = (Nu_pc + Nu_gc) * K_g / Ll

    return Coef_Molerus * zeta

# Correlação de Thanheiser (2026)
def Thanheiser(e_mf, rho_g, U_mf, U_g, Ar, rho_p, Visc_g, k_p,  K_g, cp_p, g, Pr_g, T, De, D_p, cp_g, P1, A, zeta, **kwargs):

    Ll = (Visc_g / (math.sqrt(g) * (rho_p - rho_g)))**(2/3)

    PI_2 = K_g/(2*cp_p*Visc_g)
    PI_3 = cp_g*Visc_g/K_g
    PI_4 = rho_g/(rho_p-rho_g)
    PI_5 = (U_g-U_mf)*((rho_p*cp_p/(K_g*g))**(1/3))
    PI_6 = (U_mf)*((rho_p*cp_p/(K_g*g))**(1/3))
    PI_7 = 1-e_mf
    PI_8 = De/Ll
    PI_9 = De/S1

    Nu_pc = (0.0691*PI_7/(1+PI_2*(1+0.28*(PI_7**2)*(PI_4**0.5)*PI_5*PI_6)*(1-np.exp(-6.4582E-5*PI_8))))*((1+18.9085*((PI_6/PI_5)**(1/3))*(PI_5**-1)*((1-PI_9)**-1.1523))**-1)

    Nu_gc = 0.165*((PI_3*PI_4)**(1/3))*((1+0.05*PI_6/PI_5)**-1)

    Coef_molerus_modificado = (Nu_pc + Nu_gc) * K_g / Ll

    return Coef_molerus_modificado

# Correlação de Basu (2006)
def Basu(e_mf, rho_g, U_mf, U_g, Ar, rho_p, h_gc, Visc_g, k_p,  K_g, cp_p, g, Pr_g, T, De, D_p, cp_g, P1, Tb, Fr, zeta, **kwargs):

    Cd = -180*D_p + 1.252
    Ct = 0.2007*Tb + 40.068
    x = U_g/U_mf
    Cu = 0.0067*(x**3) - 0.0857*(x**2) + 0.324*x + 0.6494
    Cpd = 0.0001*rho_p + 0.78
    Correcao = 0.78

    Coef_Basu = Correcao * Cd * Ct * Cu * Cpd

    return Coef_Basu

# Correlação de Martin (1984)
def Martin(e_mf, rho_g, U_mf, U_g, Ar, rho_p, h_gc, Visc_g, k_p,  K_g, cp_p, g, Pr_g, T, De, D_p, cp_g, P1, Fr, zeta, **kwargs):

    Gamma_25 = 0.9
    MM = 29  
    R = 8314  
    d_b = 0.015
    U_b_mean = 0.71 * ((g * d_b)**0.5)
    x = U_g - U_mf
    y = 1 - e_mf
    e = (x * y / (U_b_mean + x)) + e_mf
    p = e - e_mf
    o = 1 - e
    Z = (rho_p * cp_p / (6 * K_g)) * (((g * (D_p**3) * p) / (5 * y * o))**0.5)
    B = (1000/298.15-1)/(0.6-np.log10(1/Gamma_25-1))
    gama = (1 + 10**((0.6*B - 1 - 1000/T) / B))**-1
    K_n = (4/D_p)*(2/gama-1)*K_g*((2*math.pi*R*T/MM)
                                  ** 0.5) / (P1 * (2 * cp_g - (R / MM)))
    k = 2.6

    Nu = 4 * ((1 + K_n) * np.log(1 + 1 / K_n) - 1)

    h_pc = K_g * o * Z * (1 - np.exp(-Nu/(k*Z))) / D_p
    Coef_Martin = h_pc + h_gc

    return Coef_Martin * zeta

# Correlação de renovação de pacotes modificada por Blasczuk (2021)
def Blasczuk(e_mf, rho_g, U_mf, U_g, Ar, rho_p, h_gc, Visc_g, k_p,  K_g, cp_p, g, Pr_g, T, De, D_p, cp_g, P1, Fr, zeta, **kwargs):

    delta_b = 0.19 * (Fr**-0.23)
    t_e = 1.2*(Fr**0.3)*((D_p/De)**0.225)
    phi_b = 0.283*((k_p/K_g)**(-0.221)) # Livro do Levenspiel pelo jeito pg 325 é a espessura equivalente do filme de gás na particula e é em função do da razao de condutividade termica do gas e da particula e tambem da porosidade do leito
    e_e = 1 - ((1-e_mf)*(0.7293+(0.5139*(D_p/De)))/(1+(D_p/De)))
    k_e = e_e*K_g + ((1-e_e)*k_p*(1/(phi_b*(k_p/K_g)+(2/3))))
    rho_e = (1-e_e)*rho_p
    C_e = ((1-e_e)*cp_p*rho_p + e_e*cp_g*rho_g)/rho_e
    h_e = 2*((k_e*rho_e*C_e)**0.5)/((math.pi*t_e)**0.5)
    h_conv = (1 - delta_b)*h_e + delta_b*h_gc
    h_rad = 0
    h_b = h_conv + h_rad
    Re_e = U_g * D_p * rho_g / Visc_g
    Nu_t = 47.56*(Re_e**0.43)*(Pr_g**0.33) * \
        ((De/D_p)**(-0.74))*((C_e/cp_g)**(-1.69))
    h_t = Nu_t * k_e / De
    Coef_Blasczuk = ((2/3) * h_b) + ((1/3) * h_t)

    return Coef_Blasczuk * zeta

# Vetor de resultados
lista_resultados = []

# Condições operacionais
NF = 5  # Número de fluidização [-] #######
# U_g = 0.16 # Velocidade do gás de fluidização [m/s] #######
mf = 0.01  # vazão mássica do ar frio [kg/s]
Tf_e = 298.15  # Temperatura de entrada do fluido frio [K]
Tq_e = 673.15  # Temperatura de entrada do fluido quente [K]
P1 = 101325  # Pressão no leito [Pa]
P2 = 101325  # Pressão no tubo interno [Pa]
g = 9.807  # Gravidade [m/s²]

# Dados relacionados à construção do trocador de calor
e = 2.24/1000  # Espessura tubos de 1/4 de polegada SCH 40 [m]
De = 13.72/1000  # Diâmetro externo de um tubo [m]
Di = De-(2*e)  # Diâmetro interno de um tubo [m]
Ai_trans = math.pi*(Di**2)/4  # Área transversal interna de um tubo [m²]
Nt = 9  # Número de tubos
L = 300/1000  # Comprimento do leito [m]
W = 150/1000  # Largura do leito [m]
Ab = L*W  # Área do leito [m²]
Ai_sup = math.pi * Di * L  # Área superficial interna dos tubos [m²]
Ae_sup = math.pi * De * L  # Área superficial externa dos tubos [m²]
S1 = 0.025  # Distância longitudinal entre os tubos [m]
S2 = 0.025  # Distância transversal entre os tubos [m]
A_pt = De * L * (Nt**0.5)  # Área projetada dos tubos [m²]
A = 1 - (A_pt/Ab)  # Para cálculo da fração do tubo envoltado em bolha
T_filme = (Tq_e + Tf_e) / 2 # Temperatura de filme para cálculo da condutividade do material do tubo [K]
K_tubo = CondutividadeMaterial(T_filme) # Condutividade térmica do material na temperatura de filme [W/m*K]
zeta = (1-((De/S1)*(1+(De/(De+S2)))))**0.25 # Cálculo da correção pela disposição dos tubos

# Dados das partículas
VetorD_p = np.linspace(0.0002, 0.0004, 1000) # Vetor com variação do diâmetro da partícula [m]
rho_p = 2650  # Densidade da areia [kg/m³]
e_mf = 0.47 # Porosidade do leito em condição de mínima fluidização [-]
k_p = 1.2  # Condutividade térmica da areia [W/m*K]
cp_p = 830 # Calor específico à pressão constante da areia [J/kg*K]

# Vetor Modelos de predição do coeficiente convectivo do leito fluidizado
VetorModelo = [Molerus, Martin, Blasczuk, Basu, Thanheiser]

# Dados termofísicos dos fluidos
# Prandtl do ar na parte interna
Pr_i = PropsSI('Prandtl', 'T', Tf_e, 'P', P2, 'air')
# Densidade do ar na parte interna
rho_i = PropsSI('D', 'T', Tf_e, 'P', P2, 'air')
# Viscosidade dinâmica do ar na parte interna
Visc_i = PropsSI('V', 'T', Tf_e, 'P', P2, 'air')
# Condutividade térmica do ar na parte interna
K_i = PropsSI('L', 'T', Tf_e, 'P', P2, 'air')
# Calor específico do ar na parte interna
cp_i = PropsSI('C', 'T', Tf_e, 'P', P2, 'air')

# Prandtl do ar na parte externa
Pr_g = PropsSI('Prandtl', 'T', Tq_e, 'P', P1, 'air')
# Densidade do ar na parte externa
rho_g = PropsSI('D', 'T', Tq_e, 'P', P1, 'air')
# Viscosidade dinâmica do ar na parte externa
Visc_g = PropsSI('V', 'T', Tq_e, 'P', P1, 'air')
# Condutividade térmica do ar na parte externa
K_g = PropsSI('L', 'T', Tq_e, 'P', P1, 'air')
# Calor específico do ar na parte externa
cp_g = PropsSI('C', 'T', Tq_e, 'P', P1, 'air')

#Cálculos para a condição do escoamento interno do ar nos tubos
U_i = mf / (Ai_trans * Nt * rho_i) # Cálculo da velocidade interna [m/s]
Re_i = U_i * rho_i * Di / Visc_i # Cálculo do Reynolds interno
Nu_i, Regime = Nusselt(Pr_i, Re_i, Di)
hf = Nu_i * K_i / Di # Cálculo do coeficiente convectivo do fluido interno e do regime de escoamento interno
R1 = (1 / (hf * Ai_sup * Nt)) # Resistência térmica devido à convecção no fluido interno do tubo
Cf = mf * cp_i # Taxa de Capacidade térmica do fluido frio
R2 = (np.log(De / Di) / (2 * math.pi * K_tubo * L * Nt)) # Resistência térmica devido à condução no material do tubo

for Modelo in VetorModelo:
    Vetor_Modelo = Modelo
    for D_p in VetorD_p:

        # Número de Archimedes
        Ar = rho_g * (D_p**3) * (rho_p - rho_g) * g / (Visc_g**2)

        # Cálculo da velocidade mínima de fluidização
        U_mf = Visc_g * ((((33.7**2) + (0.0408 * Ar)) **
                          (1/2)) - 33.7) / (rho_g * D_p)

        # Cálculo da velocidade real dos fluidos e das vazões mássicas
        U_g = U_mf * NF #######
        # NF = U_g/U_mf #######
        mq = rho_g * Ab * U_g

        # Número de Froude
        Fr = D_p * g / ((U_mf**2) * ((U_g / U_mf - A)**2))

        # Coeficiente convectivo da parte do gás no leito fluidizado
        h_gc = (0.009 * (Ar**0.5) * (Pr_g**0.33)) * K_g / D_p
        
        if Modelo.__name__ == "Basu":

            Tb_ant = Tq_e
            Tb = Tb_ant
            erro = 100.0

            # Vetor Parâmetros para as funções dos modelos
            params = {"e_mf": e_mf, "rho_g": rho_g, "U_mf": U_mf, "U_g": U_g, "Ar": Ar, "rho_p": rho_p, "h_gc": h_gc, "k_p": k_p, "De": De,
                                "Visc_g": Visc_g, "K_g": K_g, "cp_p": cp_p, "g": g, "Pr_g": Pr_g, "T": Tq_e, "D_p": D_p, "Fr": Fr, "P1": P1, "cp_g": cp_g, "Tb": Tb, "A": A, "zeta": zeta}
            

            while erro > 0.1:

                # Cálculo do coeficiente convectivo do leito fluidizado borbulhante
                hq = Modelo(**params)

                R3 = (1 / (hq * Ae_sup * Nt)) # Resistência térmica devido à convecção no leito externo
                Rt = R1 + R2 + R3  # Resistência térmica total

                # Cálculo da contribuição de cada resistência térmica
                Contri_R1 = R1/Rt * 100
                Contri_R2 = R2/Rt * 100
                Contri_R3 = R3/Rt * 100

                # Cálculo da porosidade Atual
                d_b = 0.015
                U_b_mean = 0.71 * ((g * d_b)**0.5)
                e = ((U_g - U_mf) * (1 - e_mf) /
                        (U_b_mean + (U_g - U_mf))) + e_mf

                Cq = mq * cp_g # Taxa de Capacidade térmica do fluido quente (leito)

                if Cf < Cq:

                    Cmin = Cf
                    Cmax = Cq

                    # Cálculo da transferência de calor por e-NUT [W]
                    x = Cmin / Cmax
                    NUT = 1 / (Rt * Cmin)
                    e_NUT = 1 / (x + (1 / (1 - np.exp(-NUT))))
                    q_max = Cmin * (Tq_e - Tf_e)
                    q = e_NUT * q_max

                    # Cálculo das temperaturas de saída (através da conservação de energia) [K]
                    Tf_s = Tf_e + (e_NUT * (Tq_e - Tf_e))
                    Tb = Tq_e - (q / Cmax)

                else:
                    Cmin = Cq
                    Cmax = Cf

                    # Cálculo da transferência de calor por e-NUT [W]
                    x = Cmin / Cmax
                    NUT = 1 / (Rt * Cmin)
                    e_NUT = 1 / ((1 + (x * (1 / (1 - np.exp(-NUT * x))))))
                    q_max = Cmin * (Tq_e - Tf_e)
                    q = e_NUT * q_max

                    # Cálculo das temperaturas de saída (através da conservação de energia) [K]
                    Tb = Tq_e - (e_NUT * (Tq_e - Tf_e))
                    Tf_s = Tf_e + (q / Cmax)

                erro = abs(Tb_ant - Tb)

                Tb_ant = Tb

        else:

            # Vetor Parâmetros para as funções dos modelos
            params = {"e_mf": e_mf, "rho_g": rho_g, "U_mf": U_mf, "U_g": U_g, "Ar": Ar, "rho_p": rho_p, "h_gc": h_gc, "k_p": k_p, "De": De,
                                "Visc_g": Visc_g, "K_g": K_g, "cp_p": cp_p, "g": g, "Pr_g": Pr_g, "T": Tq_e, "D_p": D_p, "Fr": Fr, "P1": P1, "cp_g": cp_g, "A": A, "zeta": zeta}
            
            # Cálculo do coeficiente convectivo do leito fluidizado borbulhante
            hq = Modelo(**params)

            R3 = (1 / (hq * Ae_sup * Nt)) # Resistência térmica devido à convecção no leito externo
            Rt = R1 + R2 + R3  # Resistência térmica total

            # Cálculo da contribuição de cada resistência térmica
            Contri_R1 = R1/Rt * 100
            Contri_R2 = R2/Rt * 100
            Contri_R3 = R3/Rt * 100

            # Cálculo da porosidade Atual
            d_b = 0.015
            U_b_mean = 0.71 * ((g * d_b)**0.5)
            e = ((U_g - U_mf) * (1 - e_mf) /
                 (U_b_mean + (U_g - U_mf))) + e_mf

            Cq = mq * cp_g # Taxa de Capacidade térmica do fluido quente (leito) [W/K]

            if Cf < Cq:

                Cmin = Cf
                Cmax = Cq

                # Cálculo da transferência de calor por e-NUT [W]
                x = Cmin / Cmax
                NUT = 1 / (Rt * Cmin)
                e_NUT = 1 / (x + (1 / (1 - np.exp(-NUT))))
                q_max = Cmin * (Tq_e - Tf_e)
                q = e_NUT * q_max

                # Cálculo das temperaturas de saída (através da conservação de energia) [K]
                Tf_s = Tf_e + (e_NUT * (Tq_e - Tf_e))
                Tb = Tq_e - (q / Cmax)

            else:

                Cmin = Cq
                Cmax = Cf

                # Cálculo da transferência de calor por e-NUT [W]
                x = Cmin / Cmax
                NUT = 1 / (Rt * Cmin)
                e_NUT = 1 / ((1 + (x * (1 / (1 - np.exp(-NUT * x))))))
                q_max = Cmin * (Tq_e - Tf_e)
                q = e_NUT * q_max

                # Cálculo das temperaturas de saída (através da conservação de energia) [K]
                Tb = Tq_e - (e_NUT * (Tq_e - Tf_e))
                Tf_s = Tf_e + (q / Cmax)

# Organizar resultados para guardar todos os relevantes
        lista_resultados.append({
            "Diâmetro da Partícula": D_p,
            "Porosidade": e,
            "Velocidade Mínima de Fluidização": U_mf,
            "NUT": NUT,
            "Resistência Convecção Interna": R1,
            "Resistência Condução": R2,
            "Resistência Convecção Externa": R3,
            "Resistência Total": Rt,
            "Transferência de Calor Total [W]": q,
            "Efetividade": e_NUT,
            "Coeficiente Convectivo do Leito Fluidizado": hq,
            "Regime": Regime,
            "Modelo": Modelo.__name__,
            "Velocidade Interna": U_i,
            "Velocidade Externa": U_g,
            "Contribuição Resistência Interna": Contri_R1,
            "Contribuição Resistência Condução": Contri_R2,
            "Contribuição Resistência Externa": Contri_R3,
            "Número de Fluidização": NF,
            "Transferência de Calor Máxima": q_max
        })

df = pd.DataFrame(lista_resultados)

# DEFINIÇÃO DE CADA GRÁFICO
fig_e = px.line(
    df, x="Diâmetro da Partícula", y="Porosidade", color="Modelo",
    title="Porosidade vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Porosidade": "Porosidade"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_e.update_yaxes(range=[0, 1])
fig_e.update_layout(hovermode="x unified")

fig_U_mf = px.line(
    df, x="Diâmetro da Partícula", y="Velocidade Mínima de Fluidização", color="Modelo",
    title="Velocidade Mínima de Fluidização vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Velocidade Mínima de Fluidização": "Velocidade Mínima de Fluidização [m/s]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_U_mf.update_layout(hovermode="x unified")

fig_NUT = px.line(
    df, x="Diâmetro da Partícula", y="NUT", color="Modelo",
    title="NUT vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "NUT": "NUT"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_NUT.update_layout(hovermode="x unified")

fig_r_conv_int = px.line(
    df, x="Diâmetro da Partícula", y="Resistência Convecção Interna", color="Modelo",
    title="Resistência Convecção Interna vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Resistência Convecção Interna": "Resistência Convecção Interna [K/W]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_conv_int.update_yaxes(rangemode="nonnegative")
fig_r_conv_int.update_layout(hovermode="x unified")

fig_r_cond = px.line(
    df, x="Diâmetro da Partícula", y="Resistência Condução", color="Modelo",
    title="Resistência Condução vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Resistência Condução": "Resistência Condução [K/W]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_cond.update_yaxes(rangemode="nonnegative")
fig_r_cond.update_layout(hovermode="x unified")

fig_r_conv_ext = px.line(
    df, x="Diâmetro da Partícula", y="Resistência Convecção Externa", color="Modelo",
    title="Resistência Convecção Externa vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Resistência Convecção Externa": "Resistência Convecção Externa [K/W]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_conv_ext.update_layout(hovermode="x unified")

fig_r_total = px.line(
    df, x="Diâmetro da Partícula", y="Resistência Total", color="Modelo",
    title="Resistência Total vs. Diâmetro da Partícula (Areia)",
    labels={
        "Diâmetro da Partícula": "Diâmetro da Partícula [m]", "Resistência Total": "Resistência Total [K/W]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_total.update_layout(hovermode="x unified")

fig_q = px.line(
    df, x="Diâmetro da Partícula", y="Transferência de Calor Total [W]", color="Modelo",
    title="Transferência de Calor Total vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Transferência de Calor Total [W]": "Transferência de calor total [W]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_q.update_layout(hovermode="x unified")

fig_efetividade = px.line(
    df, x="Diâmetro da Partícula", y="Efetividade", color="Modelo",
    title="Efetividade vs. Diâmetro da Partícula (Areia)",
    labels={
        "Diâmetro da Partícula": "Diâmetro da Partícula [m]", "Efetividade": "Efetividade"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_efetividade.update_layout(hovermode="x unified")

fig_h_leito = px.line(
    df, x="Diâmetro da Partícula", y="Coeficiente Convectivo do Leito Fluidizado", color="Modelo",
    title="Coeficiente Convectivo do Leito Fluidizado vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Coeficiente Convectivo do Leito Fluidizado": "Coeficiente Convectivo do Leito [W/m²·K]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_h_leito.update_layout(hovermode="x unified")

fig_U_g = px.line(
    df, x="Diâmetro da Partícula", y="Velocidade Externa", color="Modelo",
    title="Velocidade Externa vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Velocidade Externa": "Velocidade Externa [m/s]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_U_g.update_layout(hovermode="x unified")

fig_r_contri_int = px.line(
    df, x="Diâmetro da Partícula", y="Contribuição Resistência Interna", color="Modelo",
    title="Contribuição Resistência Interna vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Contribuição Resistência Interna": "Contribuição Resistência Interna [%]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_contri_int.update_layout(hovermode="x unified")

fig_r_contri_cond = px.line(
    df, x="Diâmetro da Partícula", y="Contribuição Resistência Condução", color="Modelo",
    title="Contribuição Resistência Condução vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Contribuição Resistência Condução": "Contribuição Resistência Condução [%]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_contri_cond.update_layout(hovermode="x unified")

fig_r_contri_ext = px.line(
    df, x="Diâmetro da Partícula", y="Contribuição Resistência Externa", color="Modelo",
    title="Contribuição Resistência Externa vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Contribuição Resistência Externa": "Contribuição Resistência Externa [%]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_r_contri_ext.update_layout(hovermode="x unified")

fig_NF = px.line(
    df, x="Diâmetro da Partícula", y="Número de Fluidização", color="Modelo",
    title="Número de Fluidização vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Número de Fluidização": "Número de Fluidização"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_NF.update_layout(hovermode="x unified")

fig_q_max = px.line(
    df, x="Diâmetro da Partícula", y="Transferência de Calor Máxima", color="Modelo",
    title="Transferência de Calor Máxima vs. Diâmetro da Partícula (Areia)",
    labels={"Diâmetro da Partícula": "Diâmetro da Partícula [m]",
            "Transferência de Calor Máxima": "Transferência de Calor Máxima [W]"},
    hover_data=["Efetividade", "Transferência de Calor Total [W]", "Regime", "Coeficiente Convectivo do Leito Fluidizado"]
)
fig_q_max.update_layout(hovermode="x unified")

# CONSTRUÇÃO DA INTERFACE EM ABAS DO SITE
aba1, aba2, aba3 = st.tabs(
    ["Desempenho Global", "Hidrodinâmica & Escoamento", "Resistências Térmicas"])

with aba1:

    st.header("Desempenho Global")

    st.plotly_chart(fig_efetividade, use_container_width=True)
    st.plotly_chart(fig_q, use_container_width=True)
    st.plotly_chart(fig_q_max, use_container_width=True)
    st.plotly_chart(fig_NUT, use_container_width=True)

with aba2:
    st.header("Hidrodinâmica e Escoamento")

    st.plotly_chart(fig_h_leito, use_container_width=True)
    st.plotly_chart(fig_U_mf, use_container_width=True)
    st.plotly_chart(fig_U_g, use_container_width=True)
    st.plotly_chart(fig_NF, use_container_width=True)
    st.plotly_chart(fig_e, use_container_width=True)

with aba3:

    st.header("Resistências Térmicas")
    st.plotly_chart(fig_r_total, use_container_width=True)

    st.subheader("Contribuição das Resistências (%)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_r_contri_int, use_container_width=True)
    with col2:
        st.plotly_chart(fig_r_contri_cond, use_container_width=True)
    with col3:
        st.plotly_chart(fig_r_contri_ext, use_container_width=True)

    st.subheader("Componentes da Resistência Absoluta (K/W)")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(fig_r_conv_int, use_container_width=True)
    with col5:
        st.plotly_chart(fig_r_cond, use_container_width=True)
    with col6:
        st.plotly_chart(fig_r_conv_ext, use_container_width=True)

with st.expander("📂 Clique aqui para visualizar a tabela completa de dados"):
    st.dataframe(df)
