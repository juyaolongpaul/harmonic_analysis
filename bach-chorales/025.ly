\version "2.10.33"

\header {
  title = "25 - Wo soll ich fliehen hin"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \minor
}


soprano = \relative c' {
  \partial 4 f4 
    f4 g4 aes4 bes4 
  c2 r4 c4 
  c4. bes8 aes4 bes4 
  g2 r4 g4 
  aes4 bes4 c4 c4   %5
  bes2 c4 c4 
  aes4 bes4 c4 des8 ees8 
  c4 bes8 aes8 aes4 c4 
  ees4 c4 c4 c4 
  bes2 bes4 bes4   %10
  c4. bes8 aes4 bes8 aes8 
  g2 f4 
  
}

alto = \relative c' {
  \partial 4 c4 
    c4 d8 e8 f8 e8 f8 g8 
  e2 r4 ees4
  ees8 f8 g4~ g8 f8 e8 f8 
  e2 r4 e4 
  f4 ees4 ees4 f4   %5
  f4 ees4 ees4 ees4 
  f4 f8 g8 aes4 aes4 
  aes4 g4 ees4 aes4 
  bes4 aes4 aes4 aes4~ 
  aes4 g8 f8 g4 g4   %10
  g8 f8 e4 f2~ 
  f4 e4 c4 
  
}

tenor = \relative c' {
  \partial 4 aes8  bes8 
    c4 bes4 c4 des4 
  g,2 r4 aes4 
  aes4 g4 c4 des4 
  c2 r4 c4 
  c8 des8 c8 bes8 aes4 aes4~   %5
  aes4 g4 aes4 aes4 
  aes4 des4 c4 f4 
  ees4 des8 c8 c4 ees4 
  ees4 ees4 f4 ees4 
  f8 ees8 d4 ees4 ees8 des8   %10
  c4 c4 c4 bes8 c8 
  des4 c8 bes8 a4 
  
}

baixo = \relative c {
  \partial 4 f8  g8 
    aes4 g4 f4 des4 
  c2 r4 aes8 bes8 
  c8 d8 e4 f4 bes,4 
  c2 r4 c4 
  f4 g4 aes8 g8 f8 ees8   %5
  des4 ees4 aes,4 aes4 
  des8 c8 bes4 f'8 ees8 des4 
  ees2 aes,4 aes'4 
  g4 aes8 g8 f8 ees8 des8 c8 
  d4 bes4 ees4 g8 f8   %10
  e8 d8 c4 f8 ees8 des8 c8 
  bes4 c4 f,4 
  
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