# coding: utf-8
# Copyright (c) 2008-2011 Volvox Development Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Konstantin Lepa <konstantin.lepa@gmail.com>

"""ANSII Color formatting for output in terminal."""

from __future__ import print_function
import os


__ALL__ = [ 'colored', 'cprint' ]

VERSION = (1, 1, 0)

ATTRIBUTES = dict(
        list(zip([
            'bold',
            'dark',
            '',
            'underline',
            'blink',
            '',
            'reverse',
            'concealed'
            ],
            list(range(1, 9))
            ))
        )
del ATTRIBUTES['']


HIGHLIGHTS = dict(
        list(zip([
            'on_grey',
            'on_red',
            'on_green',
            'on_yellow',
            'on_blue',
            'on_magenta',
            'on_cyan',
            'on_white'
            ],
            list(range(40, 48))
            ))
        )


COLORS = dict(
        list(zip([
            'grey',
            'red',
            'green',
            'yellow',
            'blue',
            'magenta',
            'cyan',
            'white',
            ],
            list(range(30, 38))
            ))
        )


RESET = '\033[0m'


def colored(text, color=None, on_color=None, attrs=None):
    """Colorize text.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv('ANSI_COLORS_DISABLED') is None:
        fmt_str = '\033[%dm%s'
        if color is not None:
            text = fmt_str % (COLORS[color], text)

        if on_color is not None:
            text = fmt_str % (HIGHLIGHTS[on_color], text)

        if attrs is not None:
            for attr in attrs:
                text = fmt_str % (ATTRIBUTES[attr], text)

        text += RESET
    return text


def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Print colorize text.

    It accepts arguments of print function.
    """

    print((colored(text, color, on_color, attrs)), **kwargs)


if __name__ == '__main__':
    print('Current terminal type: %s' % os.getenv('TERM'))
    print('Test basic colors:')
    cprint('Grey color', 'grey')
    cprint('Red color', 'red')
    cprint('Green color', 'green')
    cprint('Yellow color', 'yellow')
    cprint('Blue color', 'blue')
    cprint('Magenta color', 'magenta')
    cprint('Cyan color', 'cyan')
    cprint('White color', 'white')
    print(('-' * 78))

    print('Test highlights:')
    cprint('On grey color', on_color='on_grey')
    cprint('On red color', on_color='on_red')
    cprint('On green color', on_color='on_green')
    cprint('On yellow color', on_color='on_yellow')
    cprint('On blue color', on_color='on_blue')
    cprint('On magenta color', on_color='on_magenta')
    cprint('On cyan color', on_color='on_cyan')
    cprint('On white color', color='grey', on_color='on_white')
    print('-' * 78)

    print('Test attributes:')
    cprint('Bold grey color', 'grey', attrs=['bold'])
    cprint('Dark red color', 'red', attrs=['dark'])
    cprint('Underline green color', 'green', attrs=['underline'])
    cprint('Blink yellow color', 'yellow', attrs=['blink'])
    cprint('Reversed blue color', 'blue', attrs=['reverse'])
    cprint('Concealed Magenta color', 'magenta', attrs=['concealed'])
    cprint('Bold underline reverse cyan color', 'cyan',
            attrs=['bold', 'underline', 'reverse'])
    cprint('Dark blink concealed white color', 'white',
            attrs=['dark', 'blink', 'concealed'])
    print(('-' * 78))

    print('Test mixing:')
    cprint('Underline red on grey color', 'red', 'on_grey',
            ['underline'])
    cprint('Reversed green on red color', 'green', 'on_red', ['reverse'])


# PEDRORMJUNIOR 2013/07/19:
import sys

output_color = True

if output_color:
    def grey_out(text):
        cprint(text, 'grey')

    def red_out(text):
        cprint(text, 'red')

    def green_out(text):
        cprint(text, 'green')

    def yellow_out(text):
        cprint(text, 'yellow')

    def blue_out(text):
        cprint(text, 'blue')

    def magenta_out(text):
        cprint(text, 'magenta')

    def cyan_out(text):
        cprint(text, 'cyan')

    def white_out(text):
        cprint(text, 'white')


    def grey_err(text):
        cprint(text, 'grey', 'on_grey', file=sys.stderr)

    def red_err(text):
        cprint(text, 'red', 'on_grey', file=sys.stderr)

    def green_err(text):
        cprint(text, 'green', 'on_grey', file=sys.stderr)

    def yellow_err(text):
        cprint(text, 'yellow', 'on_grey', file=sys.stderr)

    def blue_err(text):
        cprint(text, 'blue', 'on_grey', file=sys.stderr)

    def magenta_err(text):
        cprint(text, 'magenta', 'on_grey', file=sys.stderr)

    def cyan_err(text):
        cprint(text, 'cyan', 'on_grey', file=sys.stderr)

    def white_err(text):
        cprint(text, 'white', 'on_grey', file=sys.stderr)

else:
    def grey_out(text):
        print(text)

    def red_out(text):
        print(text)

    def green_out(text):
        print(text)

    def yellow_out(text):
        print(text)

    def blue_out(text):
        print(text)

    def magenta_out(text):
        print(text)

    def cyan_out(text):
        print(text)

    def white_out(text):
        print(text)


    def grey_err(text):
        print(text, file=sys.stderr)

    def red_err(text):
        print(text, file=sys.stderr)

    def green_err(text):
        print(text, file=sys.stderr)

    def yellow_err(text):
        print(text, file=sys.stderr)

    def blue_err(text):
        print(text, file=sys.stderr)

    def magenta_err(text):
        print(text, file=sys.stderr)

    def cyan_err(text):
        print(text, file=sys.stderr)

    def white_err(text):
        print(text, file=sys.stderr)

cyan_err('import termcolor')
