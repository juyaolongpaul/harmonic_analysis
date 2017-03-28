\header {
  objetivo = "dominante secundaria"
}
global = {
  \time 3/4
}
\score {
  <<
    \new Staff {
      \global
      \relative c'' {
        c d d c c b c
      }
    }
    \new Staff {
      \relative c'' {
        g a b c a g g
      }
    }
    \new Staff {
      \global
      \relative c' {
        e f g e fis f e
      }
    }
    \new Staff {
      \global
      \relative c' {
        c b e a, d g, c
      }
    }
  >>
}