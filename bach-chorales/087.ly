\version "2.10.33"

\header {
  title = "87 - Du, o schönes Weltgebäude"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key g \minor
}

soprano = \relative c {
  \repeat volta 2 {
    r4 c''2 g4 
    c c bes bes8 a 
    a2 g 
    r4 ees2 g4 
    f ees8 d ees4 d 
    c1 
  }
  \repeat volta 2 {
    ees2 f4 g 
    aes aes g g 
    f2 ees 
  }
  g4 a bes g 
  c c b2 
  c4 ees d c 
  c b c2 
}


alto = \relative c {
  \repeat volta 2 {
    r4 ees'2 d4 
    fis8 g a4 a8 g g4 
    g fis d2 
    r4 c2 c4 
    d8 b c4 c b 
    g1 
  }
  \repeat volta 2 {
    c2 d4 ees 
    f f f8 d ees4 
    ees d bes2 
  }
  ees4 ees f ees 
  ees ees8 d d2 
  g4 g aes8 g4 f8 
  g4 g g2 
}


tenor = \relative c {
  \repeat volta 2 {
    r4 g'2 bes4 
    c ees8 d d4 ees 
    d4. c8 b2 
    r4 c2 g4 
    aes a g g8 f 
    ees1 
  }
  \repeat volta 2 {
    g2 bes 
    c4 bes bes c 
    c bes8 aes g2 
  }
  bes4 c bes bes 
  c8 bes a4 g2 
  g4 c c8 b c d 
  ees4 d e2 
}


baixo = \relative c {
  \repeat volta 2 {
    r4 c2 bes4 
    a fis' g c, 
    d2 g, 
    r4 aes'2 ees4 
    f fis g g, 
    c1 
  }
  \repeat volta 2 {
    c'4 bes aes g 
    f d ees c 
    aes bes ees2 
  }
  ees4 c d ees 
  aes8 g fis4 g2 
  ees8 d c4 f8 g aes4 
  g g, c2 
}




\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}