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
- MSSubClass
  - カテゴリ変数に変換
- LotArea
  - 対数化
- YearRemodAdd
  - is_remodという変数を追加
    - if YearRemodAdd != YearBuilt
  - PastYearsという変数を追加
    - this year - max(YearBuilt, YearRemodAdd)
- MasVnrArea
  - 対数化
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
