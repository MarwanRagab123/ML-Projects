# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. تحميل البيانات من المسار الصحيح
data = pd.read_csv('e:/Intern 2/Task1/data/Mall_Customers (1).csv')

# 2. تحويل النوع الجنسي
data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})

# 3. اختيار الأعمدة الرقمية فقط للتجميع
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 4. تطبيق KMeans مع معالجة الأخطاء
try:
    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    
    # 5. تصور النتائج وحفظها
    plt.figure(figsize=(10,6))
    plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                s=300, c='red', marker='*', label='مراكز التجميع')
    plt.xlabel('الدخل السنوي (ألف $)')
    plt.ylabel('معدل الإنفاق (1-100)')
    plt.title('تجميع العملاء باستخدام K-Means')
    plt.legend()
    plt.grid(True)
    
    # حفظ الصورة
    plt.savefig('e:/Intern 2/Task2/clusters_result.png', dpi=300, bbox_inches='tight')
    print('تم حفظ نتائج التجميع في:\n"e:/Intern 2/Task2/clusters_result.png"')
    
    # عرض معلومات التجميع
    print('\nمعلومات التجميع:')
    print(f'- عدد المجموعات: {kmeans.n_clusters}')
    print(f'- عدد العملاء في كل مجموعة: {pd.Series(y_kmeans).value_counts().to_string()}')
    
    # عرض الصورة
    plt.show()
    
except Exception as e:
    print(f"حدث خطأ: {e}")
    print("تأكد من:")
    print("- صحة مسار ملف البيانات")
    print("- تثبيت جميع المكتبات المطلوبة")
    print("- أن البيانات تحتوي على الأعمدة المطلوبة")