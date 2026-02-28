import numpy as np

def calculate_motor_geometry(
    thrust_N, 
    burn_time_s, 
    chamber_pressure_bar, 
    ambient_pressure_bar=1.01325, 
    thrust_coefficient=1.5,   # Varsayılan değerler
    c_star=1500.0,            # Varsayılan değerler
    fixed_OF_ratio=5.0, 
    gamma=1.2, 
    fuel_density=900.0,       # Parafin vb. için ortalama
    regression_coeff_a=0.0001, 
    regression_exp_n=0.5, 
    initial_port_radius_mm=20.0
):
    """
    Hedeflenen itki ve süreye göre motorun fiziksel boyutlarını hesaplar.
    """
    
    # --- Birim Dönüşümleri ---
    chamber_pressure_Pa = chamber_pressure_bar * 1e5
    ambient_pressure_Pa = ambient_pressure_bar * 1e5
    port_radius_m = initial_port_radius_mm / 1000.0
    
    # --- Nozzle / Boğaz Hesabı ---
    # F = Cf * At * Pc
    throat_area = thrust_N / (thrust_coefficient * chamber_pressure_Pa)
    throat_diameter_m = np.sqrt(4 * throat_area / np.pi)
    
    # --- Kütle Akış Hesabı ---
    # m_dot = Pc * At / c*
    total_mass_flow = chamber_pressure_Pa * throat_area / c_star
    oxidizer_mass_flow = total_mass_flow / (1 + 1 / fixed_OF_ratio)
    
    # --- İteratif Yanma Odası Hesabı ---
    tolerance = 1e-4
    grain_length_m = 0.0
    web_thickness_m = 0.0
    final_regression_rate = 0.0
    
    # İstenen O/F oranını tutturana kadar çapı genişleterek dener
    for _ in range(1000):
        port_area = np.pi * port_radius_m**2
        oxidizer_mass_flux = oxidizer_mass_flow / port_area
        
        # Yanma Hızı: r = a * G^n
        regression_rate = regression_coeff_a * oxidizer_mass_flux**regression_exp_n
        
        fuel_mass_flow = oxidizer_mass_flow / fixed_OF_ratio
        
        # Yakıt Uzunluğu Hesabı
        grain_length_m = fuel_mass_flow / (
            fuel_density * regression_rate * 2 * np.pi * port_radius_m
        )
        
        # O/F Kontrolü
        OF_calculated = oxidizer_mass_flow / (
            fuel_density * regression_rate * 2 * np.pi * port_radius_m * grain_length_m
        )
        
        if abs(OF_calculated - fixed_OF_ratio) < tolerance:
            final_regression_rate = regression_rate
            break
        
        # Yaklaşamazsa port yarıçapını güncelle
        port_radius_m *= (1 + 0.1 * (OF_calculated - fixed_OF_ratio))
        
    # --- Sonuç Geometrisi ---
    web_thickness_m = final_regression_rate * burn_time_s
    outer_radius_m = port_radius_m + web_thickness_m
    
    # Sözlük (Dictionary) olarak döndür ki her yerde kullanalım
    return {
        "throat_diameter_mm": throat_diameter_m * 1000,
        "grain_length_mm": grain_length_m * 1000,
        "inner_diameter_mm": 2 * port_radius_m * 1000,
        "outer_diameter_mm": 2 * outer_radius_m * 1000,
        "oxidizer_mass_flow": oxidizer_mass_flow,
        "burn_time": burn_time_s
    }

# --- Test Etmek İçin ---
if __name__ == "__main__":
    # Dosyayı direkt çalıştırırsan burası çalışır
    sonuc = calculate_motor_geometry(
        thrust_N=100.0, 
        burn_time_s=8.0, 
        chamber_pressure_bar=10.0
    )
    
    print("\n==== HESAPLANAN MOTOR ====")
    for key, val in sonuc.items():
        print(f"{key}: {val:.2f}")