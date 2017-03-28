\version "2.10.33"

\header {
  title = "299 - Meinen Jesum laß ich nicht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c'' {
  bes4 bes4 c4 c4 
  d4 d8 ees16 f16 ees2 
  f4 f4 ees4 ees4 
  d4 c8 d16 ees16 c2 
  bes2 bes4 bes4   %5
  c4 c4 bes4 aes8 g8 
  g2 bes4 bes4 
  aes4 aes4 g4 f8 g16 aes16 
  f2 ees2 
  bes'4 bes4 c4 c4   %10
  d4 d8 ees16 f16 ees2 
  bes4 bes8 c16 des16 c8 bes8 aes8 g8 
  f8 g16 aes16 f4 ees2 
  
}

alto = \relative c'' {
  g4 g4 ees4 aes4 
  aes8 g8 aes4 g2 
  bes4 bes8 aes8 g4. a8 
  bes8 a8 g8 ees8 f4. ees8 
  d2 ees4 ees4   %5
  ees4 aes4 aes8 g8 f4 
  e2 e8 f8 g8 e8 
  c4 f8 d8 ees4 ees4 
  ees4 d4 bes2 
  g'4 g4 aes4 aes8 g8   %10
  f4 g4 g2 
  ees4 ees4 ees8 d8 ees4 
  ees4 d4 bes2 
  
}

tenor = \relative c' {
  ees4 ees4 c4 f4 
  f8 ees8 f8 d8 bes2 
  d8 ees8 f8 d8 d8 c16 b16 c4 
  bes4 bes4 bes8 g8 a4 
  f2 g8 aes8 bes8 g8   %5
  aes8 bes8 c4 f,8 g8 aes8 bes8 
  c2 bes8 aes8 g4 
  aes4 bes4 bes4 c8 aes8 
  bes8 g8 aes4 g2 
  ees'4 ees4 ees4 f4   %10
  b,8 c8 d8 b8 c2 
  g4 g4 aes8 f8 ees8 bes'8 
  c4 bes8 aes8 g2 
  
}

baixo = \relative c {
  ees8 f8 g8 ees8 aes8 g8 aes8 f8 
  bes4 bes,4 ees2 
  bes8 c8 d8 bes8 c8 d8 ees8 f8 
  g8 f8 ees8 c8 f4 f,4 
  bes2 ees8 f8 g8 ees8   %5
  aes8 g8 f8 ees8 d8 e8 f4 
  c2 g'8 f8 e8 c8 
  f8 ees8 d8 bes8 ees8 g8 aes8 f8 
  bes4 bes,4 ees2 
  ees'8 des8 c8 bes8 aes8 g8 f8 ees8   %10
  d8 c8 b8 g8 c2 
  ees8 des8 c8 bes8 aes8 bes8 c8 bes8 
  aes4 bes4 ees2 
  
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