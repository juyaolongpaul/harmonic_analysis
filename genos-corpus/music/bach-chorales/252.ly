\version "2.10.33"

\header {
  title = "252 - Jesu, nun sei gepreiset"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key bes \major
}

soprano = \relative c {
 \repeat volta 2{
  \partial 4 d''4 
  d bes c d 
  ees d8 c c4 c 
  c bes bes8 g a4 
  bes2. c4 
  c c c8 d16 ees d8 c 
  bes2 bes4 bes 
  a8 f bes ees d c c4 
  bes2. } a4 
  a a a a 
  bes2 bes4 bes 
  a a a a 
  bes2 bes4 d 
  d8 c bes4 c d 
  ees2 d4 d 
  c8 b c f ees d d4 
  \time 3/4
  c2 b4 
  c2 c4 
  d2 d4 
  ees2. 
  d2 d4 
  c2 c4 
  c ees8 d c b 
  c2 b4 
  c2 c4 
  d2 d4 
  ees2. 
  d2 d4 
  c2 bes4 
  bes d8 c bes a 
  bes2. 
}


alto = \relative c {
  \repeat volta 2 {
  \partial 4 f'4 
  g8 fis g4 g8 a bes4 
  bes8 a bes4 a g 
  fis d g f 
  f2. f8 e 
  f g a g a4. fis8 
  d ees f4 ees g 
  f8 ees d ees f g f ees 
  d2. } c4 
  f f fis8 g a fis 
  g ees f4 ees g 
  f ees8 d c4 f 
  f ees d f8 g 
  a4 d, g8 f f4 
  ees8 d c f f4 f 
  f8 d ees f g aes g f 
  \time 3/4
  ees2 f4 
  g2 f8 ees 
  d4 d'8 c bes aes 
  g f ees g c bes 
  a4 d, g 
  g f ees8 f 
  g4 aes g 
  g2 g4 
  g f2~ 
  f8 g aes4 g 
  g2 f4 
  f2 g4 
  g2 g4 
  f g f 
  f2. 
}


tenor = \relative c {
  \repeat volta 2 {
  \partial 4 bes'4 
  bes8 c d4 c f 
  ees8 c f4 f c 
  d8 a bes4 c c 
  d2. a8 bes 
  c d ees4 d a 
  bes aes g c 
  c bes bes a 
  f2. } f8 a 
  c4 c d d8 c 
  bes4 aes g c 
  c c f c 
  bes8 aes g4 f bes 
  a bes8 a g a bes4 
  bes a bes bes 
  c c c b 
  \time 3/4
  g2 d'4 
  c bes aes 
  a2 bes4(
  b) c2~
  c4 bes4 bes 
  bes aes8 bes c d 
  ees4 f d 
  ees2 d4 
  c f ees 
  d8 ees f4 ees8 d 
  c4 bes c
  bes2 bes4 
  c d ees 
  d ees c 
  d2. 
}


baixo = \relative c {
  \repeat volta 2 {
  \partial 4 bes'8 a 
  g a g f ees f ees d 
  c4 d8 ees f4 ees 
  d g8 f e4 f 
  bes,2. f'8 g 
  a bes c4 fis,4. d8 
  g f ees d ees4 e 
  f g f8 ees f4 
  bes,2. } f4 
  f'8 g f ees d e fis d 
  g f ees d ees4 e 
  f c8 d ees g f ees 
  d bes ees4 bes bes' 
  fis g8 f ees f ees d 
  c4 f bes, bes' 
  a aes g8 f g4 
  \time 3/4
  c,2 d4 
  ees e f
  f fis g 
  g aes g 
  fis g f 
  e f8 g aes4 
  g f g 
  c,2 g'4 
  aes a2 
  bes4 b2 
  c4 g a 
  bes8 c bes a g f 
  ees g f ees d c 
  f4 ees f 
  bes,2. 
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