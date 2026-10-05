
from sklearn.base import BaseEstimator, TransformerMixin


class TicketGroupSizeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.ticket_counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        X = X.copy()

        X["TicketGroupSize"] = (
            X["Ticket"]
            .map(self.ticket_counts_)
            .fillna(1)
        )

        return X
