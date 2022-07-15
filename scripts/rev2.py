# fairness F(u) => (0, 1)
# value G(p) => (-1, 1)
# ratings liability (given user) R(u, p) => (0, 1)
# initially, F(u) = G(p) = R(u, p) = 1
import pandas as pd


def linear_map_to_range(rating, min_value, max_value, min_range, max_range, rating_mapping=None):
    """
    Linear transformation of any rating into [min_range, max_range] range

    :param rating: rating of something (product, review, etc) in stars (ONE, TWO, ... FIVE) OR fp number
    :param min_value: minimum numerical representation of rating
    :param max_value: maximum numerical representation of rating
    :param min_range: minimum value in the new range
    :param max_range: maximum value in the new range
    :param rating_mapping: a map that converts a string rating into a numeric rating
    :return: FP value between min_range and max_range representing that mapping
    """
    value = rating_mapping[rating] if rating_mapping is not None else rating
    return (value - min_value) / (max_value - min_value) * (max_range - min_range) + min_range


def main():
    amazon_reviews = pd.read_csv('amazon_ratings.csv',
                                 header=None,
                                 names=['userID', 'productID', 'rating', 'timestamp'],
                                 usecols=['userID', 'productID', 'rating'])


    gamma_1, gamma_2 = 0.5, 0.5
    amazon_reviews['rating'] = amazon_reviews['rating'].apply(linear_map_to_range, args=(1, 5, -1, 1))
    amazon_reviews['r'] = 1

    amazon_reviews['ones'] = 1

    amazon_reviews['userDegree'] = amazon_reviews.groupby('userID')['rating'].transform('count')
    amazon_reviews['productDegree'] = amazon_reviews.groupby('productID')['rating'].transform('count')

    # rev 2
    # vectorized in order to speed up operations
    iterations = 20
    for _ in range(iterations):
        # f
        amazon_reviews['f_users'] = amazon_reviews.groupby('userID')['r'].transform('sum') / amazon_reviews[
            'userDegree']

        # g
        amazon_reviews['r_rating_product'] = amazon_reviews[['r', 'rating']].product(axis=1)
        amazon_reviews['g_users'] = amazon_reviews.groupby('productID')['r_rating_product'].transform('sum') / \
                                    amazon_reviews['productDegree']

        # r
        amazon_reviews['r'] = ((amazon_reviews['ones'] - (
            ((amazon_reviews['rating'] - amazon_reviews['g_users']).abs()).div(2))).multiply(
            gamma_2) + amazon_reviews['f_users'].multiply(gamma_1)).multiply(1 / (gamma_1 + gamma_2))

    # aquellos con mas de 5 reviews y con un f_usuario <= 0.2 -> malos
    bad_users = amazon_reviews[
        (amazon_reviews.groupby('userID')['rating'].transform('count') >= 5) & (amazon_reviews['f_users'] <= 0.2)]

    bad_users.to_csv('bad_users.csv')

    # aquellos con mas de 10 reviews y con un f_usuario >= 0.9 -> extremadamente justos
    extremely_fair = amazon_reviews[
        (amazon_reviews.groupby('userID')['rating'].transform('count') >= 10) & (amazon_reviews['f_users'] >= 0.9)]

    extremely_fair.to_csv('extremely_fair.csv')


if __name__ == '__main__':
    main()
