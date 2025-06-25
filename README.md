# Customer-Experience-Data-Analysis-Patika-Bootcamp-Project1

## 📦 Dataset Description

- **Kaynak:** [Kaggle - Customer Experience Dataset](https://www.kaggle.com/datasets/ziya07/customer-experience-dataset/data)
- **Amaç:** Müşteri deneyimini, memnuniyetini ve elde tutma (retention) oranını anlamak ve modellemek için kullanılabilecek, simüle edilmiş müşteri verisi.
- **Boyut:** 1000 satır, 14 sütun.

## 🗂️ Kolonlar ve Temel Açıklama

- **Customer_ID:** Unique müşteri kimliği (1-1000, modelleme için kullanılmaz)
- **Age:** Müşteri yaşı (18-69)
- **Gender / Gender_Encoded:** Müşteri cinsiyeti (Female/Male ve 0/1 olarak kodlanmış)
- **Location / Location_Encoded:** Müşteri lokasyonu (Kentsel/Banliyö/Kırsal ve 0/1/2 kodlu)
- **Num_Interactions:** Platformdaki toplam etkileşim sayısı
- **Feedback_Score:** 1-5 arası müşteri geri bildirimi
- **Products_Purchased:** Satın alınan ürün sayısı (1-19)
- **Products_Viewed:** Görüntülenen ürün sayısı (5-49)
- **Time_Spent_on_Site:** Sitede geçirilen toplam süre (5-60 dakika)
- **Satisfaction_Score:** 1-10 arası memnuniyet puanı
- **Retention_Status / Retention_Status_Encoded:** Elde tutuldu mu (0: kayıp, 1: korundu)
- Diğer: Analiz sırasında oluşturulan kategorik segmentler.

---

## 📊 Veri Analizi Adımları

### 1. Veri Setinin Yüklenmesi ve Tanıtımı

```python
df = pd.read_csv("customer_experience_data.csv")
df.head()
df.tail()
print(df.info())
print(df.columns.tolist())
print(df.shape)
```

- **Sonuç:** Veri setinde 1000 satır ve 14 sütun var, veri yapısı beklendiği gibi.

### 2. Kolon Türleri ve Özet İstatistikler

```python
print(df.dtypes)
summary_stats = df.describe(include='all').T
display(summary_stats)
```

- **Sonuç:** 10 integer, 1 float, 3 object tipi değişken var.
- **ID benzersiz, Age ortalama 44, Satın Alma ortalama 10, Feedback ve Satisfaction dağılımı geniş.**

### 3. Detaylı Değişken İstatistikleri & İçgörü

- **Yaş:** Ortalama 43,8 – Genç ve yaşlı segmentlerde çift modlu dağılım.
- **Etkileşim:** Medyan ve ortalama yakın, sağa hafif çarpık.
- **Feedback:** En çok 3 puan verilmiş, simetrik dağılım.
- **Satın Alma:** 5-15 arası ürün yoğun, hafif sağa çarpık.
- **Görüntüleme:** 25-30 arası zirve, simetrik.
- **Süre:** 15-20 ve 45-50 dakikada iki tepe, farklı davranış paternleri var.
- **Memnuniyet:** 3 ve 7-8 puanlarında yoğunluk; çoğunluk orta memnuniyetli.
- **Cinsiyet/Lokasyon:** Dengeli.

### 4. Eksik Değer Analizi

```python
print(df.isnull().sum())
```

- **Sonuç:** Eksik veri bulunmamaktadır.
- **İçgörü:** Ek temizleme veya doldurma işlemine ihtiyaç yok.

### 5. Aykırı Değer Analizi

- Boxplot’lar ile tüm sayısal değişkenler incelendi, **belirgin aykırı değer yok**.
- **Veri istatistiksel olarak dengeli ve temiz.**

---

## 🛠️ Veri Manipülasyonu ve Segmentasyon

### 1. Cinsiyete Göre Retention

```python
retention_by_gender = df.groupby('Gender_Encoded')['Retention_Status_Encoded'].mean()
```

- Erkeklerin retention oranı (%70), kadınlardan (%68) bir miktar daha yüksek.

### 2. Feedback Skoruna Göre Memnuniyet & Retention

```python
feedback_stats = df.groupby('Feedback_Score').agg({
    'Satisfaction_Score': 'mean',
    'Retention_Status_Encoded': 'mean'
})
```

- Feedback ile retention/memnuniyet arasında **güçlü korelasyon yok**.

### 3. Etkileşim Seviyesine Göre Segmentasyon

```python
df['Interaction_Level'] = pd.cut(df['Num_Interactions'], bins=[-1, 3, 7, 15, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])
```

- Etkileşim arttıkça retention da artıyor.

### 4. Satın Alma Segmentlerine Göre Retention

```python
df['Purchased_Segment'] = pd.cut(df['Products_Purchased'], bins=[-1, 0, 2, 5, np.inf], labels=['None', 'Few', 'Moderate', 'Many'])
```

- **‘Few’ segmentinde retention oranı en yüksek**, çok ürün alanlarda biraz daha düşük.

### 5. Memnuniyet Seviyesine Göre Retention

```python
df['Satisfaction_Level'] = pd.cut(df['Satisfaction_Score'], bins=[-np.inf, 3, 6, 8, 10], labels=['Low', 'Medium', 'High', 'Very High'])
```

- Memnuniyet seviyesi arttıkça retention oranı da yükseliyor.

### 6. Sitede Geçirilen Süre Quartile Analizi

```python
df['Time_Spent_Quartile'] = pd.qcut(df['Time_Spent_on_Site'], 4, labels=['Q1','Q2','Q3','Q4'])
```

- Sitede geçirilen süre ile retention arasında net bir doğrusal ilişki yok.

### 7. Ürün Görüntüleme/Satın Alma Davranışı Segmentasyonu

```python
df['View_Purchase_Ratio'] = df['Products_Purchased'] / (df['Products_Viewed']+1)
df['Engagement_Level'] = pd.cut(df['View_Purchase_Ratio'], bins=[-np.inf, 0.1, 0.3, 0.6, 1], labels=['Very Low', 'Low', 'Medium', 'High'])
```

- Engagement arttıkça retention artıyor.

### 8. Lokasyonlara Göre Retention

```python
top_locations = df['Location_Encoded'].value_counts().head(5).index
df[df['Location_Encoded'].isin(top_locations)].groupby('Location_Encoded')['Retention_Status_Encoded'].mean()
```

- Bazı lokasyonlarda retention oranı anlamlı derecede yüksek.

---

## 📈 Korelasyon ve İkili (Bivariate) Analizler

### Korelasyon Matrisi:

- Korelasyonlar çok düşük (max ±0.06); **güçlü doğrusal ilişki yok**.

### Bivariate Analiz (Örnekler):

- **Products_Purchased ↔ Feedback_Score**
- **Products_Viewed ↔ Products_Purchased**
- **Time_Spent_on_Site ↔ Products_Purchased**
- **Products_Purchased ↔ Satisfaction_Score**
- **Num_Interactions ↔ Products_Purchased**

---

## 🖼️ Veri Görselleştirme

- **Sayısal Değişken Dağılımları:** Histogram + KDE
- **Kategorik Değişken Dağılımları:** Barplot, heatmap
- **Memnuniyet ve Retention ilişkisi:** Segment bazında kutu grafikleri, scatter plot
- **Korelasyon Isı Haritası:** Tüm sayısal değişkenler için.
- **Ek Görseller:**

  - Retention by Satisfaction, by Interactions, by Product Segment
  - Gender/Location vs Retention heatmap
  - Segmentasyon ve davranış kategorilerinin görselleştirilmesi

---

## 📌 Sonuçlar & Özet İçgörüler

- **Veri Temiz ve Dengeil:** Eksik ya da aykırı değer yok.
- **Retention’da Lokasyon, Etkileşim, Memnuniyet Etkili:** En yüksek retention yüksek memnuniyet, yüksek etkileşim, bazı lokasyonlarda ve düşük alışverişte gözlendi.
- **Doğrudan Güçlü Korelasyon Yok:** İlişkiler çok değişkenli, dolayısıyla makine öğrenmesi için çok boyutlu modelleme önerilir.
- **İş Fırsatları:** Segment ve davranış kırılımına göre kişiselleştirilmiş kampanya ve iletişim ile retention artışı mümkün.

---

## 📂 Proje Adımları (Özet Akış)

1. Veri yükleme & genel inceleme
2. Eksik & aykırı değer analizi
3. Temel istatistiksel özet
4. Kategorik/sayısal segmentasyon & veri manipülasyonu
5. Korelasyon & bivariate analiz
6. Gelişmiş görselleştirme
7. Özet içgörüler ve iş önerileri

---

