require('@tensorflow/tfjs-node');   //اجرای tfjs روی cpu لپ تاپ
const tf = require('@tensorflow/tfjs'); //در Tensorflow JS lib مورد نیاز است
const loadCSV = require('./load-csv');  //لود فایل csv

function knn(features, labels, predictionPoint, k) {
    const {mean, variance} = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))    //هزینه پیش بینی مقیاس شده برای استانداردسازی

return (
   features
/* Step 0 - استاندارد سازی ویژگی ها برای  */
    .sub(mean)
    .div(variance.pow(0.5))
/* Step 1 - فاصله بین ویژگی‌ها و ویژگی‌های نقطه پیش‌بینی را پیدا کنید */
    .sub(scaledPrediction)	//عملیات پخش
    .pow(2)	//عملیات Elementwise برای مربع هر عنصر
    .sum(1) //مجموع در امتداد محور x(1).
    .sqrt()	//عملیات Elementwise برای گرفتن توان 0.5 = sqrt هر عنصر
/* Step 2 - مرتب سازی از کمترین فاصله به بزرگترین فاصله */
    .expandDims(1) //ابعاد تانسور فواصل را در سرتاسر محور x گسترش می دهیم تا شکل [4،1] را به دست آوریم، همان فاصله برچسب ها
    .concat(labels, 1) //ما برچسب ها را به فواصل در سراسر محور x متصل می کنیم تا با شاخص های مشابه در یک تانسور به هم مرتبط شوند.
    .unstack() //ما 1 تانسور خود را در 1 آرایه وانیلی JS که حاوی چندین تانسور است، جدا می کنیم
/* پس از برداشتن تانسور، با آرایه وانیلی JS سروکار داریم و از این مرحله به بعد فقط می‌توانیم از روش‌های وانیلی JS استفاده کنیم.*/
    .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1) //مرتب سازی تابع tp sprt تانسورها به ترتیب فاصله حداقل تا بیشترین
    /* Step 3 - میانگین ارزش برچسب رکوردهای k بالا */ 
    .slice(0, k)//Get Top k records
    .reduce((acc, pair) => acc + pair.get(1), 0) / k //Get average label value
)}
//فراخوانی تابع CSV بارگذاری شده با فایل CSV ارسال شده و انجام برخی پیش پردازش روی مجموعه داده
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {  
    shuffle: true, //ردیف های داده را در فایل CSV به هم بزنید تا مجموعه داده آزمایشی تصادفی شود
    splitTest: 10, //تقسیم داده های آزمون در 2 مجموعه داده (10 برای تست / 20000+ دیگر برای آموزش)
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],   //Features - Latitude/Longitude/SqFt Lot/SqFt_Living
    // sqft_living متراژ مربع فضای نشیمن داخلی در یک خانه
    // sqft_lot متراژ مربع زمین یا زمینی که خانه روی آن ساخته شده
    labelColumns: ['price']       //Label- قیمت خانه (به هزار دلار)
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('KNN Housing Price Prediction: $', result);     //خروج از سیستم پیش بینی مسکن
    console.log();  //New Line
    console.log('KNN Prediction Error Percentage: %', err * 100);    //درصد خطا خروج از سیستم     
    console.log();  //New Line

})
