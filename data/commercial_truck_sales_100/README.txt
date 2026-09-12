Commercial truck completed-sale dataset

100 completed Purple Wave sales: 50 box trucks and 50 semi tractors.
Each record includes six gallery images sampled across the original listing.
truck_sales_100.csv has one row per truck; image paths use | as the separator.
raw_json preserves the public listing data used to create each row.

sale_price_usd_including_buyer_premium is the displayed purchase price.
winning_bid_usd excludes the buyer premium. Taxes and transport are excluded.

eBay was evaluated, but automated access to completed listings triggered a
verification challenge. Standard eBay marketplace access did not provide a
reliable bulk export of completed commercial-truck transactions, so no eBay
records were represented as verified sold-price labels in this version.
