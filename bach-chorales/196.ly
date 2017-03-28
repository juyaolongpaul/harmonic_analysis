\version "2.10.33"

\header {
  title = "196 - Da der Herr Christ zu Tische saß"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 bes4 g4 
  bes4 b4 c4 d4 
  ees8 d8 c8 d8 ees4 d4 
  c4 c4 b4 g4 
  c8 d8 ees4 d4 c4   %5
  b2 c4 g4 
  g4 a4 bes4 aes4 
  g4 f4 ees4 ees4 
  bes'4 bes4 c4 d4 
  ees4 d4 ees4 ees4   %10
  f4 ees4 d4 c4 
  c4 b4 c
  
}

alto = \relative c' {
  \partial 4 ees8  f8 
    g8 f8 ees4 f4. ees8 
  g4 f4 ees4 g4 
  g4 g4 g4 g4 
  g4 g8 f8 g4 d4 
  g4 g4 aes8 g8 g8 f8   %5
  g8 aes8 g8 f8 ees4 ees8 f8 
  g4 f4 f8 g4 f8~ 
  f8 ees4 d8 ees4 c4 
  ees8 f8 g4 ees4 f4 
  g4 f4 g4 g8 aes8~   %10
  aes8 g8 g4 f8 aes8 g8 f8 
  ees8 d16 c16 d8 f~ f e
  
}

tenor = \relative c' {
  \partial 4 c4 
    d4. c8 bes4 bes4 
  bes8 c8 d4 g,4 d'4 
  c8 d8 ees8 d8 c4 b4 
  c8 b8 c4 d4 b4 
  c4 c4. b8 c4   %5
  d4 g,4 g4 c4 
  c8 g8 c4 d8 c4 d16 c16 
  bes8 c8 f,8 bes16 aes16 g4 g8 aes8 
  bes8 aes8 bes8 des8 c8 bes8 aes4 
  bes4 bes4 bes4 c4   %10
  d4. c4 b8 c8 aes8 
  g2 g4
  
}

baixo = \relative c {
  \partial 4 c4 
    b4 c4 d4 ees4~ 
  ees4 d4 c4 b4 
  c2~ c8 ees8 g8 f8 
  ees4 aes4 g4 g8 f8 
  ees8 d8 c4 f8 g8 aes4   %5
  g4 g,4 c4 c8 d8 
  ees4. f16 ees16 d8 ees8 f4 
  g8 aes8 bes8 bes,8 c4 c4 
  g8 f8 ees4 aes8 g8 f4 
  ees8 g8 bes4 ees4 c4~   %10
  c8 b8 c4 d4 ees8 f8 
  g4 g,4 c
  
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