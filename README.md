# RecPadroes-Voz

graph TD
    subgraph Power Supply
        USB_5V --- R1(10k)
        R1 --- V_REF(Virtual Ground @ 2.5V)
        V_REF --- R2(10k)
        R2 --- GND
    end

    subgraph Input Stage
        Piezo_Positive --- C1(1µF)
        Piezo_Negative --- GND
        C1 --- D1(Zener) & D2(Zener)
        D1 --- Non_Inv_Input(Pin 3)
        D2 --- GND
        subgraph Diode Protection
            D1 --- D2
        end
    end

    subgraph Amplifier
        USB_5V --- VCC(Pin 8)
        GND --- VEE(Pin 4)

        Non_Inv_Input --- TL082(U1A)
        TL082 --- Output(Pin 1)
        Inv_Input(Pin 2) --- R3(10k)
        R3 --- GND
        Inv_Input --- RV1(100k Trimpot)
        RV1 --- Output
        V_REF --- Non_Inv_Input
    end

    subgraph Output
        Output --- Amplified_Out
    end

    style V_REF fill:#f9f,stroke:#333,stroke-width:2px
