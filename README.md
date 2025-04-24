# Chihuahua_or_Muffin
Disclaimer: Muffins and chihuahuas were not harmed in the making of this model.

## Accuracy
| Epochs | Accuracy |
|:------:|---------:|
| 1      | 86.4%    |
| 2      | 87.5%    |
| 3      | 89.8%    |
| 4      | 90.6%    |
| 5      | 92.1%    |
| 6      | 92.1%    |
| 7      | 91.8%    |
| _**8**_      | _**92.3%**_    |
| 9      | 90.6%    |
| 10     | 90.4%    |

![Image](https://github.com/user-attachments/assets/c45d2ac3-8eee-467d-beea-6b8475c299f5)
(Not so smart yet)

## Dataset
For the image, we used the Muffin vs chihuahua dataset from Kaggle.
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

print("Path to dataset files:", path)
```
All images have been resized to 128x128, ready for CNN's tasty munchies (?).

## Sample Predictions
|       Input     |   Prediction  |
|:---------------:|:-------------:|
| Dog-like Muffin | Chihuahua ✅  |
| Muffin-like Dog | Muffin ✅     |
| Mystery         | You decide 🤯 |

## License
MIT. Go forth and classify.
