\version "2.10.33"

\header {
  title = "323 - Wie schön leuchtet der Morgenstern"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key f \major
}

soprano = \relative c' {
  \repeat volta 2 {
  \partial 4 f4 
  c'8 bes a g f4 c' 
  d d c4 c 
  d4 e f e 
  d d c4 a
  d4 c bes a 
  g2 f4 } r4 
  c'2 a 
  c a2 
  a4 a g g 
  a a g g 
  a a g2 
  f2 f'4 e4 
  d c bes a 
  g2 f4 
}


alto = \relative c' {
  \repeat volta 2 {
  \partial 4 c4 
    c c d8 e f4 
    f f f4 g 
    a4 g f8 g g4 
    f f e4 f 
    f4 e f8 g f4 
    f e f4 } r4 
  g2 a 
  g f2 
  f4 f4 e8 d c4 
  c c8 d e4 e 
  f f f e 
  f2 a4 g4 
  f e d c 
  d c c4 
}


tenor = \relative c {
  \repeat volta 2 {
  \partial 4 a'4 
    g f8 g a bes c4 
    c8 bes16 a bes8 bes a4 c
    c4 b c8 d e g, 
    a d b4 c4 c 
    bes8 a g4 d'8 c c4 
    c2 c4 } r4 
  c2 c 
  c c2 
  c4 c c c8 bes 
  a bes c4 c c 
  c d d c8 bes 
  a2 d8 c8 bes4 
  bes8 a g4 f8 g a c 
  bes8. a16 g4 a4 
}


baixo = \relative c {
  \repeat volta 2 {
  \partial 4 f4 
    e f8 e d4 a 
    bes8 c d e f4 e 
    f4 g a8 b c e, 
    f d g4 c,4 f 
    bes,4 c d8 e f bes, 
    c2 f,4 } r4 
  e'2 f 
  c f,2 
  f8 g8 a bes c d e c 
  f g a bes c bes a g 
  f e d c bes g c4 
  f,2 d8 f8 g a 
  bes4 c d8 e f a, 
  bes g c4 f,4 
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