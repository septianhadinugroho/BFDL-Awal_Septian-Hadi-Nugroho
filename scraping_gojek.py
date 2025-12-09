"""
Scraping Google Play Store - Aplikasi Gojek
Target: 10.000+ reviews dengan 3 kelas sentimen
"""

from google_play_scraper import Sort, reviews, app
import pandas as pd
import time
from datetime import datetime

def scrape_gojek_reviews(target_count=12000):

    print("=" * 60)
    print("SCRAPING GOJEK REVIEWS - GOOGLE PLAY STORE")
    print("=" * 60)
    
    app_id = 'com.gojek.app'
    
    all_reviews = []
    continuation_token = None
    
    print(f"\n🎯 Target: {target_count} reviews")
    print("📱 App: Gojek Indonesia\n")
    
    # Get app info
    try:
        app_info = app(app_id, lang='id', country='id')
        print(f"📊 App Info:")
        print(f"   - Nama: {app_info['title']}")
        print(f"   - Rating: {app_info['score']}")
        print(f"   - Total Reviews: {app_info['reviews']}")
        print(f"   - Installs: {app_info['installs']}\n")
    except Exception as e:
        print(f"⚠️  Warning: Tidak bisa ambil app info: {e}\n")
    
    batch_num = 0
    
    # Scrape dengan batch sampai mencapai target
    while len(all_reviews) < target_count:
        batch_num += 1
        try:
            print(f"🔄 Batch {batch_num}: Mengambil reviews...")
            
            # Ambil reviews (max 200 per batch)
            result, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token
            )
            
            if not result:
                print("⚠️  Tidak ada review lagi yang bisa diambil")
                break
            
            all_reviews.extend(result)
            print(f"   ✅ Berhasil ambil {len(result)} reviews")
            print(f"   📈 Total sekarang: {len(all_reviews)} reviews\n")
            
            # Delay untuk menghindari rate limit
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error pada batch {batch_num}: {e}")
            print("⏸️  Menunggu 5 detik sebelum retry...\n")
            time.sleep(5)
            
            # Jika sudah cukup data, break
            if len(all_reviews) >= target_count * 0.8:
                print("ℹ️  Data sudah cukup, menghentikan scraping")
                break
    
    # Convert ke DataFrame
    print("\n" + "=" * 60)
    print("PROCESSING DATA")
    print("=" * 60)
    
    df = pd.DataFrame(all_reviews)
    
    # Pilih kolom yang diperlukan
    df_clean = df[['userName', 'score', 'content', 'at', 'thumbsUpCount']].copy()
    df_clean.columns = ['username', 'rating', 'review', 'date', 'thumbs_up']
    
    # Convert date
    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')
    
    # Bersihkan data
    df_clean = df_clean[df_clean['review'].notna()].copy()
    df_clean = df_clean[df_clean['review'].str.strip() != ''].copy()
    df_clean = df_clean[df_clean['review'].str.len() >= 10].copy()  # Min 10 karakter
    
    # Labeling sentimen berdasarkan rating
    def label_sentiment(rating):
        if rating <= 2:
            return 'negatif'
        elif rating == 3:
            return 'netral'
        else:  # 4 atau 5
            return 'positif'
    
    df_clean['sentiment'] = df_clean['rating'].apply(label_sentiment)
    
    # Hapus duplikat
    df_clean = df_clean.drop_duplicates(subset=['review']).copy()
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"\n📊 STATISTIK DATA:")
    print(f"   - Total reviews dikumpulkan: {len(df_clean)}")
    print(f"\n   Distribusi Sentimen:")
    sentiment_counts = df_clean['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   - {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\n   Distribusi Rating:")
    rating_counts = df_clean['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"   - {rating} bintang: {count}")
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'gojek_reviews_{timestamp}.csv'
    df_clean.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Data berhasil disimpan ke: {filename}")
    
    # Save juga dalam JSON
    json_filename = f'gojek_reviews_{timestamp}.json'
    df_clean.to_json(json_filename, orient='records', force_ascii=False, indent=2)
    print(f"✅ Data JSON disimpan ke: {json_filename}")
    
    print("\n" + "=" * 60)
    print("SCRAPING SELESAI!")
    print("=" * 60)
    
    return df_clean, filename

def balance_dataset(df, target_per_class=3500):

    print("\n" + "=" * 60)
    print("BALANCING DATASET")
    print("=" * 60)
    
    from sklearn.utils import resample
    
    # Pisahkan per kelas
    df_positif = df[df['sentiment'] == 'positif']
    df_negatif = df[df['sentiment'] == 'negatif']
    df_netral = df[df['sentiment'] == 'netral']
    
    print(f"\n📊 Distribusi Awal:")
    print(f"   - Positif: {len(df_positif)}")
    print(f"   - Negatif: {len(df_negatif)}")
    print(f"   - Netral: {len(df_netral)}")
    
    # Oversample atau undersample ke target
    df_positif_balanced = resample(df_positif, 
                                   replace=len(df_positif) < target_per_class,
                                   n_samples=target_per_class,
                                   random_state=42)
    
    df_negatif_balanced = resample(df_negatif,
                                   replace=len(df_negatif) < target_per_class,
                                   n_samples=target_per_class,
                                   random_state=42)
    
    df_netral_balanced = resample(df_netral,
                                  replace=len(df_netral) < target_per_class,
                                  n_samples=target_per_class,
                                  random_state=42)
    
    # Gabungkan
    df_balanced = pd.concat([df_positif_balanced, df_negatif_balanced, df_netral_balanced])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Distribusi Setelah Balancing:")
    sentiment_counts = df_balanced['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_balanced)) * 100
        print(f"   - {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\n✅ Total data setelah balancing: {len(df_balanced)}")
    
    return df_balanced

if __name__ == "__main__":
    # Scrape data
    df, filename = scrape_gojek_reviews(target_count=12000)
    
    # Balance dataset jika perlu
    if len(df) >= 10000:
        print("\n🎯 Data mencukupi! Melakukan balancing...")
        df_balanced = balance_dataset(df, target_per_class=3500)
        
        # Save balanced dataset
        balanced_filename = filename.replace('.csv', '_balanced.csv')
        df_balanced.to_csv(balanced_filename, index=False, encoding='utf-8-sig')
        print(f"\n✅ Balanced dataset disimpan ke: {balanced_filename}")
    else:
        print(f"\n⚠️  Hanya berhasil mengumpulkan {len(df)} reviews")
        print("💡 Tips: Coba scrape lagi atau gunakan app lain sebagai tambahan")