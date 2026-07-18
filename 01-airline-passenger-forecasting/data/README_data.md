# Dataset Notes

The included `airline_passengers_sample.csv` is the classic monthly international airline-passenger series commonly known as **AirPassengers**.

- **Rows:** 144 monthly observations
- **Period:** January 1949 through December 1960
- **Columns:** `Month`, `Passengers`
- **Passenger unit:** thousands of passengers
- **Purpose in this repository:** reproducible demonstration data for time-series preprocessing, LSTM training, evaluation, and deployment

Source used by the original notebook:

```text
https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

## Expected uploaded CSV format

```csv
Month,Passengers
2019-01-01,112000
2019-02-01,118000
2019-03-01,132000
```

The app automatically:

- parses monthly dates,
- sorts chronologically,
- aggregates duplicate months using a sum,
- inserts missing months,
- interpolates missing monthly passenger values,
- rejects negative passenger counts.

For meaningful forecasts, use at least **24 consecutive months** of history. A longer history is strongly preferred.
