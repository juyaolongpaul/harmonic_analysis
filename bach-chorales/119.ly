\version "2.10.33"

\header {
  title = "119 - Es woll uns Gott genädig sein 1"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c' {
  \partial 4 f8  g8 
    aes4 bes4 c4 bes4 
  ees4 d8 c8 c4 bes4 
  ees4 d4 c4 bes4 
  aes4 g4 f4 f8 g8 
  aes4 bes4 c4 bes4   %5
  ees4 d8 c8 c4 bes4 
  ees4 d4 c4 bes4 
  aes4 g4 f4 f4 
  bes4 bes4 g4 c4 
  c4 b4 c4 c4   %10
  f4 f4 g8 f8 ees4 
  f8 ees8 d4 c4 d4 
  ees4 d4 c4 bes4 
  c8 bes8 aes4 g4 f8 g8 
  aes4 bes4 c8 bes8 aes4   %15
  bes8 aes8 g4 f4 f'4 
  f4 f4 g4 f4 
  ees4 d4 c2 
  
}

alto = \relative c' {
  \partial 4 d4 
    c8 d8 ees4 ees8 f8 g4 
  g4 g4 g4 g4 
  g4 g4 aes8 g8 f4 
  c8 f4 e8 c4 c4 
  f4 g4 aes4 g4   %5
  c4 b4 g4 g4 
  g8 c8 c8 bes8 bes8 aes8 g8 e8 
  f4 e4 c4 c4 
  f4 ees8 d8 ees8 f8 g4 
  g4 g4 g4 g4   %10
  c4 bes4 bes4 c8 bes8 
  aes4 d,8 ees16 f16 ees4 g4 
  g4 f4 ees8 f8 g4 
  c,4 f4 e4 c4 
  f4 g4 g8 e8 f4   %15
  g8 f8 e4 c4 c'4 
  bes4 bes4 bes4 c8 f,8 
  g4. f8 e2 
  
}

tenor = \relative c' {
  \partial 4 bes4 
    aes4 g4 aes4 ees'8 d8 
  c4 b4 ees4 d4 
  c4 d8 e8 f4 f,8 g8 
  aes8 bes8 c4 aes4 aes8 g8 
  f8 aes8 des4 ees4 ees8 f8   %5
  g4 g4 ees4 d4 
  c4 f,8 g8 aes4 bes4 
  c4 bes8 aes8 aes4 aes8 g8 
  f4 bes4 bes4 ees4 
  d8 c8 d4 ees4 ees4   %10
  ees4 d4 ees8 d8 c4 
  d8 c8 b4 g4 b4 
  c8 bes8 aes8 bes8 c8 d8 e8 f8 
  g4 c,4 c4 aes8 g8 
  f8 aes8 des4 c4 des4   %15
  des4 g,8 aes16 bes16 aes4 f4 
  d'8 ees8 f8 d8 ees8 d8 c4 
  c4 b4 g2 
  
}

baixo = \relative c {
  \partial 4 bes4 
    f'4 ees4 aes4 g8 f8 
  ees8 f8 g4 c,4 g'4 
  c4 bes4 aes4 d,8 e8 
  f4 c4 f,4 f'8 ees8 
  des8 c8 bes4 aes4 ees'8 d8   %5
  c4 g'4 c4 g4 
  c,4 d8 e8 f4 des4 
  c8 bes8 c4 f,4 f'8 ees8 
  d8 c8 bes4 ees4 d8 c8 
  g'4 g,4 c4 c'8 bes8   %10
  a4 bes4 ees,4 aes8 g8 
  f4 g4 c,4 g'4 
  c,4 f8 g8 aes4 g8 f8 
  e4 f4 c4 f8 ees8 
  des8 c8 bes4 aes4 des8 c8   %15
  bes4 c4 f,4 a'4 
  bes8 c8 d8 bes8 ees,4 aes4 
  g8 f8 g8 g,8 c2 
  
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