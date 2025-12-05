FEE_RATIO = 0.01


def take_order(order):
    if order.is_test:
        return

    if order.is_expired:
        return

    if order.state != 'WAITING':
        return

    if order.coupon:
        discount_rate = order.coupon.discount_rate
        order.price *= 1 - discount_rate

    order.price = sum(item.price for item in order.items)
    fee = order.price * FEE_RATIO

    order.fee = fee

    order.state = 'TAKEN'

    start_delivery(order)


def start_delivery(order):
    pass


def calculate_result(x, y, z, w):
    dist_xy = x ** 2 + y ** 2
    dist_zw = z ** 2 + w ** 2

    if dist_xy < dist_zw:
        val = z * w - x * y
    else:
        val = x * y - z * w

    tot_sum = x + y + z + w

    return val * tot_sum
