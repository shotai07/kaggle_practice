### Categorical
- Street
  - 関係なさそう？
- LotConfig
  - 関係なさそう？
- RoofMatl
  - 変更なし
  - ひょっとしたらWdShnglか否かが効きそう
- ExterQual
  - 変更なし
  - is_Exが効きそう
- BsmtQual
  - 変更なし
  - is_Exが効きそう
- KichenQual
  - is_Exが効きそう

### Numerical
- MSSubClass, OverallQual, OverallCond
  - カテゴリ変数に変換
- LotArea, LotFrontage
  - 対数化
  - LotFrontageの欠損値：median埋め
- YearRemodAdd
  - is_remodという変数を追加
    - if YearRemodAdd != YearBuilt
  - PastYearsという変数を追加
    - this year - max(YearBuilt, YearRemodAdd)
- MasVnrArea
  - 対数化
  - 欠損値：mode埋め
- GarageYrBlt
  - 欠損値：平均値埋め
- BsmtFinSf2
  - 使わない
- HalfBath
  - 使わない
- KitchenAbvGr
  - 使わない
- YrSold
  - 使わない

### others
- index=1298は外れ値のため、使わない
  - BsmtSF1>5000だったり、TotalBsmtSF>6000だったりと外れ値。
