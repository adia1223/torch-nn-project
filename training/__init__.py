def pretty_time(seconds):
    hours = seconds // 3600
    minutes = seconds // 60 % 60
    seconds %= 60
    res = ""
    if hours:
        res += '%d h' % hours
        res += ' '
        res += '%d m' % minutes
        res += ' '
        res += '%d s' % int(seconds)
    elif minutes:
        res += '%d m' % minutes
        res += ' '
        res += '%d s' % int(seconds)
    else:
        res += '%.2f s' % seconds
    return res
